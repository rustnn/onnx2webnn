/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared parsing and selection for the transformers.js model manifest.

use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Entry {
    pub file: String,
    /// Immutable Hugging Face revision for reproducible downloads.
    #[serde(default)]
    pub revision: Option<String>,
    /// Expected SHA-256 of the primary ONNX file.
    #[serde(default)]
    pub sha256: Option<String>,
    #[serde(default)]
    pub heavy: bool,
    /// Reason this model cannot build on the CoreML backend.
    #[serde(default)]
    pub coreml_unsupported: Option<String>,
    /// Reason this model is prohibitively slow to build on the CoreML backend.
    #[serde(default)]
    pub coreml_slow: Option<String>,
    #[serde(default)]
    pub override_dims: HashMap<String, u32>,
    #[serde(default)]
    pub pin_inputs: HashMap<String, i64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Selection {
    All,
    Match(String),
}

impl Selection {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "all" => Ok(Self::All),
            value if value.starts_with("match=") && value.len() > "match=".len() => {
                Ok(Self::Match(value["match=".len()..].to_string()))
            }
            _ => Err(format!(
                "invalid selection {value:?}: expected all or match=<text>"
            )),
        }
    }

    pub fn includes(&self, index: usize, entry: &Entry) -> bool {
        match self {
            Self::All => true,
            Self::Match(needle) => entry.label(index).contains(needle),
        }
    }
}

impl Entry {
    pub fn revision(&self) -> &str {
        self.revision.as_deref().unwrap_or("main")
    }

    pub fn source_key(&self) -> String {
        format!(
            "{}@{}#{}",
            self.file,
            self.revision(),
            self.sha256.as_deref().unwrap_or("")
        )
    }

    pub fn label(&self, index: usize) -> String {
        format!(
            "#{index} {} dims={:?} pins={:?}",
            self.file, self.override_dims, self.pin_inputs
        )
    }

    /// Stable cache identity. Map entries are sorted before hashing, so two
    /// cases using the same file but different overrides or pins cannot clash.
    pub fn cache_key(&self) -> String {
        let mut canonical = format!("webnn-cache-v1\nfile={}\n", self.file);
        if let Some(revision) = &self.revision {
            canonical.push_str(&format!("revision={revision}\n"));
        }
        if let Some(sha256) = &self.sha256 {
            canonical.push_str(&format!("sha256={sha256}\n"));
        }
        let mut dims: Vec<_> = self.override_dims.iter().collect();
        dims.sort_unstable_by_key(|(name, _)| *name);
        for (name, value) in dims {
            canonical.push_str(&format!("dim={name}:{value}\n"));
        }
        let mut pins: Vec<_> = self.pin_inputs.iter().collect();
        pins.sort_unstable_by_key(|(name, _)| *name);
        for (name, value) in pins {
            canonical.push_str(&format!("pin={name}:{value}\n"));
        }
        let hash = canonical.bytes().fold(0xcbf29ce484222325u64, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
        });
        let stem = Path::new(&self.file)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model")
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
            .collect::<String>();
        format!("{stem}-{hash:016x}")
    }
}

pub fn manifest_path() -> PathBuf {
    std::env::var_os("O2W_MANIFEST")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/models/manifest.json"))
}

pub fn load_manifest() -> Result<Vec<Entry>, String> {
    load_manifest_from(&manifest_path())
}

pub fn load_manifest_from(path: &Path) -> Result<Vec<Entry>, String> {
    let text =
        std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    parse_manifest(&text).map_err(|e| format!("parse {}: {e}", path.display()))
}
pub fn parse_manifest(text: &str) -> Result<Vec<Entry>, String> {
    let entries: Vec<Entry> = serde_json::from_str(text).map_err(|e| e.to_string())?;
    for (index, entry) in entries.iter().enumerate() {
        if entry.file.is_empty() {
            return Err(format!("manifest entry #{index} has an empty file"));
        }
        if let Some(revision) = &entry.revision {
            if revision.is_empty()
                || !revision
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
            {
                return Err(format!(
                    "manifest entry #{index} has an invalid revision {revision:?}"
                ));
            }
        }
        if let Some(sha256) = &entry.sha256 {
            if entry.revision.is_none() {
                return Err(format!(
                    "manifest entry #{index} has sha256 without an immutable revision"
                ));
            }
            if !valid_sha256(sha256) {
                return Err(format!(
                    "manifest entry #{index} has an invalid lowercase sha256"
                ));
            }
        }
    }
    Ok(entries)
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn entry() -> Entry {
        serde_json::from_str(
            r#"{"file":"org--repo/onnx/model.onnx","override_dims":{"b":1},"pin_inputs":{"branch":0}}"#,
        )
        .unwrap()
    }

    #[test]
    fn parses_and_selects_manifest_cases() {
        let entry = entry();
        assert!(Selection::All.includes(0, &entry));
        assert!(Selection::Match("model.onnx".into()).includes(0, &entry));
        assert!(!Selection::Match("missing".into()).includes(0, &entry));
    }

    #[test]
    fn cache_key_is_stable_and_configuration_specific() {
        let first = entry();
        let reordered: Entry = serde_json::from_str(
            r#"{"pin_inputs":{"branch":0},"override_dims":{"b":1},"file":"org--repo/onnx/model.onnx"}"#,
        )
        .unwrap();
        assert_eq!(first.cache_key(), reordered.cache_key());

        let mut changed = first.clone();
        changed.override_dims.insert("b".into(), 2);
        assert_ne!(first.cache_key(), changed.cache_key());
        changed = first.clone();
        changed.pin_inputs.insert("branch".into(), 1);
        assert_ne!(first.cache_key(), changed.cache_key());

        changed = first.clone();
        changed.revision = Some("0123456789abcdef0123456789abcdef01234567".into());
        changed.sha256 =
            Some("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".into());
        assert_ne!(first.cache_key(), changed.cache_key());
    }

    #[test]
    fn validates_revision_and_digest() {
        assert!(
            parse_manifest(
                r#"[{"file":"org--repo/model.onnx","sha256":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"}]"#
            )
            .unwrap_err()
            .contains("without an immutable revision")
        );
        assert!(
            parse_manifest(r#"[{"file":"org--repo/model.onnx","revision":"../main"}]"#)
                .unwrap_err()
                .contains("invalid revision")
        );
        assert!(
            parse_manifest(
                r#"[{"file":"org--repo/model.onnx","revision":"0123456789abcdef0123456789abcdef01234567","sha256":"ABCDEF"}]"#
            )
            .unwrap_err()
            .contains("invalid lowercase sha256")
        );
    }

    #[test]
    fn rejects_removed_validation_metadata() {
        let error =
            parse_manifest(r#"[{"file":"org--repo/model.onnx","validation":{"tier":"blocked"}}]"#)
                .unwrap_err();
        assert!(error.contains("unknown field"));
    }

    #[test]
    fn curated_ci_manifest_is_small_pinned_and_unique() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/models/ci-validation.json");
        let entries = load_manifest_from(&path).expect("load curated CI manifest");
        assert!(!entries.is_empty(), "CI manifest must not be empty");
        let mut cases = HashSet::new();
        for entry in entries {
            assert!(!entry.heavy, "CI model {} must not be heavy", entry.file);
            let revision = entry.revision.as_deref().expect("CI model revision");
            assert!(
                revision.len() == 40
                    && revision
                        .bytes()
                        .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f')),
                "CI model {} must use a full immutable commit SHA",
                entry.file
            );
            assert!(
                entry.sha256.as_deref().is_some_and(valid_sha256),
                "CI model {} must declare its SHA-256",
                entry.file
            );
            assert!(
                cases.insert(entry.cache_key()),
                "duplicate CI model case for {}",
                entry.file
            );
        }
    }

    #[test]
    fn selector_rejects_tiers_and_malformed_values() {
        assert!(Selection::parse("smoke").is_err());
        assert!(Selection::parse("extended").is_err());
        assert!(Selection::parse("quick").is_err());
        assert!(Selection::parse("match=").is_err());
    }
}
