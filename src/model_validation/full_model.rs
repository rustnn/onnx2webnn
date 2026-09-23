/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Full Hugging Face model downloads for the manual numerical sweep.

use super::manifest::Entry;
use crate::protos::onnx::{GraphProto, ModelProto};
use huggingface_hub::{HFClient, HFClientSync, HFRepositorySync, RepoDownloadFileParams};
use prost::Message;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs;
use std::io::Read;
use std::path::{Component, Path, PathBuf};

const CACHE_FORMAT: u32 = 3;
const METADATA_DIR: &str = ".onnx2webnn-validation";

#[derive(Debug, Deserialize, Serialize)]
struct CacheMetadata {
    format: u32,
    repository: String,
    revision: String,
    sha256: Option<String>,
    primary: String,
    files: Vec<CachedFile>,
}

#[derive(Debug, Deserialize, Serialize)]
struct CachedFile {
    path: String,
    length: u64,
}

pub fn cache_root() -> Result<PathBuf, String> {
    resolve_cache_root(
        nonempty_env("O2W_ONNX_CACHE"),
        nonempty_env("O2W_CACHE_DIR"),
        nonempty_env("HF_HUB_CACHE"),
        nonempty_env("HUGGINGFACE_HUB_CACHE"),
        nonempty_env("HF_HOME"),
        dirs::cache_dir(),
    )
    .ok_or_else(|| {
        "cannot determine the Hugging Face cache directory; set O2W_ONNX_CACHE or HF_HUB_CACHE"
            .to_string()
    })
}

fn resolve_cache_root(
    o2w_onnx: Option<PathBuf>,
    o2w_shared: Option<PathBuf>,
    hf_hub: Option<PathBuf>,
    legacy_hf_hub: Option<PathBuf>,
    hf_home: Option<PathBuf>,
    os_cache: Option<PathBuf>,
) -> Option<PathBuf> {
    o2w_onnx
        .or_else(|| o2w_shared.map(|path| path.join("onnx")))
        .or(hf_hub)
        .or(legacy_hf_hub)
        .or_else(|| hf_home.map(|path| path.join("hub")))
        .or_else(|| os_cache.map(|path| path.join("huggingface").join("hub")))
}

fn nonempty_env(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

pub fn cache_full_model(entry: &Entry) -> Result<PathBuf, String> {
    let cache_root = cache_root()?;
    fs::create_dir_all(&cache_root).map_err(|e| format!("create {}: {e}", cache_root.display()))?;
    let revision = entry.revision();
    let (repository, repository_path) = parse_manifest_file(&entry.file)?;
    let (owner, name) = repository
        .split_once('/')
        .ok_or_else(|| format!("invalid Hugging Face repository {repository:?}"))?;
    let client = HFClient::builder()
        .cache_dir(&cache_root)
        .build()
        .map_err(|e| format!("create Hugging Face client: {e}"))?;
    let client = HFClientSync::from_api(client)
        .map_err(|e| format!("create blocking Hugging Face client: {e}"))?;
    let repository_client = client.model(owner, name);
    let metadata_path = metadata_path(&cache_root, entry);
    let refresh = std::env::var_os("O2W_MODEL_CACHE_REFRESH").is_some();

    if !refresh {
        if let Some(model_path) = complete_cache(
            &repository_client,
            &metadata_path,
            &repository,
            revision,
            entry.sha256.as_deref(),
        )? {
            return Ok(model_path);
        }
    }

    if metadata_path.exists() {
        fs::remove_file(&metadata_path)
            .map_err(|e| format!("remove stale {}: {e}", metadata_path.display()))?;
    }

    let mut model_path = download_hub_file(
        &repository_client,
        &repository_path,
        revision,
        refresh,
        false,
    )?;
    if let Some(expected) = entry.sha256.as_deref() {
        if let Err(error) = verify_sha256(&model_path, expected) {
            if refresh {
                return Err(error);
            }
            model_path =
                download_hub_file(&repository_client, &repository_path, revision, true, false)?;
            verify_sha256(&model_path, expected)?;
        }
    }

    let model_bytes =
        fs::read(&model_path).map_err(|e| format!("read {}: {e}", model_path.display()))?;
    let model = ModelProto::decode(model_bytes.as_slice())
        .map_err(|e| format!("decode {}: {e}", model_path.display()))?;
    let locations = external_locations(&model)?;
    let repository_parent = Path::new(&repository_path)
        .parent()
        .unwrap_or_else(|| Path::new(""));
    let mut files = vec![cached_file(&repository_path, &model_path)?];
    for location in locations {
        let repository_sidecar = repository_parent.join(&location);
        let repository_sidecar = path_for_url(&repository_sidecar)?;
        let sidecar_path = download_hub_file(
            &repository_client,
            &repository_sidecar,
            revision,
            refresh,
            false,
        )?;
        files.push(cached_file(&repository_sidecar, &sidecar_path)?);
    }

    let metadata = CacheMetadata {
        format: CACHE_FORMAT,
        repository,
        revision: revision.to_string(),
        sha256: entry.sha256.clone(),
        primary: repository_path,
        files,
    };
    write_metadata(&metadata_path, &metadata)?;
    Ok(model_path)
}

fn download_hub_file(
    repository: &HFRepositorySync,
    path: &str,
    revision: &str,
    force_download: bool,
    local_files_only: bool,
) -> Result<PathBuf, String> {
    repository
        .download_file(
            &RepoDownloadFileParams::builder()
                .filename(path)
                .revision(revision)
                .force_download(force_download)
                .local_files_only(local_files_only)
                .build(),
        )
        .map_err(|e| format!("download {path} at revision {revision}: {e}"))
}

fn cached_file(repository_path: &str, local_path: &Path) -> Result<CachedFile, String> {
    let length = fs::metadata(local_path)
        .map_err(|e| format!("inspect {}: {e}", local_path.display()))?
        .len();
    Ok(CachedFile {
        path: repository_path.to_string(),
        length,
    })
}

fn parse_manifest_file(file: &str) -> Result<(String, String), String> {
    let (org_repo, relative) = file
        .split_once('/')
        .ok_or_else(|| format!("{file}: expected <org>--<repo>/<path>"))?;
    let (org, repo) = org_repo
        .split_once("--")
        .ok_or_else(|| format!("{file}: expected <org>--<repo>/<path>"))?;
    if org.is_empty() || repo.is_empty() {
        return Err(format!(
            "{file}: empty Hugging Face organization or repository"
        ));
    }
    let relative = path_for_url(&safe_relative_path(relative)?)?;
    Ok((format!("{org}/{repo}"), relative))
}

fn safe_relative_path(path: impl AsRef<Path>) -> Result<PathBuf, String> {
    let path = path.as_ref();
    if path.as_os_str().is_empty() || path.is_absolute() {
        return Err(format!("unsafe external-data path '{}'", path.display()));
    }
    let mut clean = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => clean.push(part),
            _ => return Err(format!("unsafe external-data path '{}'", path.display())),
        }
    }
    Ok(clean)
}

fn path_for_url(path: &Path) -> Result<String, String> {
    let safe = safe_relative_path(path)?;
    Ok(safe
        .components()
        .map(|component| component.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/"))
}

fn metadata_path(cache_root: &Path, entry: &Entry) -> PathBuf {
    let digest = Sha256::digest(entry.source_key().as_bytes());
    let key = digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    cache_root
        .join(METADATA_DIR)
        .join(format!("{key}.complete.json"))
}

fn complete_cache(
    repository_client: &HFRepositorySync,
    metadata_path: &Path,
    repository: &str,
    revision: &str,
    expected_sha256: Option<&str>,
) -> Result<Option<PathBuf>, String> {
    let Ok(bytes) = fs::read(metadata_path) else {
        return Ok(None);
    };
    let Ok(metadata) = serde_json::from_slice::<CacheMetadata>(&bytes) else {
        return Ok(None);
    };
    if !metadata_matches(&metadata, repository, revision, expected_sha256) {
        return Ok(None);
    }

    let mut primary = None;
    for file in &metadata.files {
        let path = match path_for_url(Path::new(&file.path)) {
            Ok(path) => path,
            Err(_) => return Ok(None),
        };
        let local_path = match download_hub_file(repository_client, &path, revision, false, true) {
            Ok(path) => path,
            Err(_) => return Ok(None),
        };
        let valid_length = fs::metadata(&local_path)
            .map(|value| value.is_file() && value.len() == file.length)
            .unwrap_or(false);
        if !valid_length {
            return Ok(None);
        }
        if path == metadata.primary {
            primary = Some(local_path);
        }
    }

    let Some(primary) = primary else {
        return Ok(None);
    };
    if expected_sha256.is_some_and(|expected| verify_sha256(&primary, expected).is_err()) {
        return Ok(None);
    }
    Ok(Some(primary))
}

fn metadata_matches(
    metadata: &CacheMetadata,
    repository: &str,
    revision: &str,
    expected_sha256: Option<&str>,
) -> bool {
    metadata.format == CACHE_FORMAT
        && metadata.repository == repository
        && metadata.revision == revision
        && metadata.sha256.as_deref() == expected_sha256
        && !metadata.primary.is_empty()
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 1 << 20];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|e| format!("hash {}: {e}", path.display()))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn verify_sha256(path: &Path, expected: &str) -> Result<(), String> {
    let actual = sha256_file(path)?;
    if actual == expected {
        Ok(())
    } else {
        Err(format!(
            "SHA-256 mismatch for {}: expected {expected}, got {actual}",
            path.display()
        ))
    }
}

fn write_metadata(path: &Path, metadata: &CacheMetadata) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
    }
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("complete.json");
    let part = path.with_file_name(format!("{file_name}.part"));
    let bytes = serde_json::to_vec_pretty(metadata).map_err(|e| e.to_string())?;
    fs::write(&part, bytes).map_err(|e| format!("write {}: {e}", part.display()))?;
    if path.exists() {
        fs::remove_file(path).map_err(|e| format!("replace {}: {e}", path.display()))?;
    }
    fs::rename(&part, path)
        .map_err(|e| format!("move {} to {}: {e}", part.display(), path.display()))
}

fn external_locations(model: &ModelProto) -> Result<Vec<PathBuf>, String> {
    fn walk(graph: &GraphProto, locations: &mut BTreeSet<PathBuf>) -> Result<(), String> {
        for tensor in &graph.initializer {
            if tensor.data_location != 1 {
                continue;
            }
            let location = tensor
                .external_data
                .iter()
                .find(|entry| entry.key == "location")
                .map(|entry| entry.value.as_str())
                .ok_or_else(|| format!("external tensor '{}' has no location", tensor.name))?;
            locations.insert(safe_relative_path(location)?);
        }
        for node in &graph.node {
            for attribute in &node.attribute {
                if let Some(graph) = &attribute.g {
                    walk(graph, locations)?;
                }
                for graph in &attribute.graphs {
                    walk(graph, locations)?;
                }
            }
        }
        Ok(())
    }

    let mut locations = BTreeSet::new();
    if let Some(graph) = &model.graph {
        walk(graph, &mut locations)?;
    }
    Ok(locations.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protos::onnx::{StringStringEntryProto, TensorProto};

    #[test]
    fn model_cache_precedence_prefers_o2w_then_hugging_face() {
        let path = resolve_cache_root(
            Some("/o2w-models".into()),
            Some("/o2w".into()),
            Some("/hf-hub".into()),
            Some("/legacy-hf-hub".into()),
            Some("/hf-home".into()),
            Some("/os-cache".into()),
        );
        assert_eq!(path, Some(PathBuf::from("/o2w-models")));

        let path = resolve_cache_root(
            None,
            Some("/o2w".into()),
            Some("/hf-hub".into()),
            None,
            None,
            None,
        );
        assert_eq!(path, Some(PathBuf::from("/o2w/onnx")));

        let path = resolve_cache_root(
            None,
            None,
            Some("/hf-hub".into()),
            None,
            Some("/hf-home".into()),
            Some("/os-cache".into()),
        );
        assert_eq!(path, Some(PathBuf::from("/hf-hub")));

        let path = resolve_cache_root(None, None, None, None, None, Some("/os-cache".into()));
        assert_eq!(path, Some(PathBuf::from("/os-cache/huggingface/hub")));
    }

    #[test]
    fn cache_metadata_requires_current_format_revision_and_digest() {
        let metadata = CacheMetadata {
            format: CACHE_FORMAT,
            repository: "org/repo".into(),
            revision: "0123456789abcdef0123456789abcdef01234567".into(),
            sha256: Some("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".into()),
            primary: "onnx/model.onnx".into(),
            files: Vec::new(),
        };
        let digest = metadata.sha256.as_deref();
        assert!(metadata_matches(
            &metadata,
            &metadata.repository,
            &metadata.revision,
            digest
        ));
        assert!(!metadata_matches(
            &metadata,
            &metadata.repository,
            "other-revision",
            digest
        ));
        assert!(!metadata_matches(
            &metadata,
            &metadata.repository,
            &metadata.revision,
            None
        ));

        let mut old = metadata;
        old.format -= 1;
        assert!(!metadata_matches(
            &old,
            &old.repository,
            &old.revision,
            old.sha256.as_deref()
        ));
    }

    #[test]
    fn computes_sha256_for_download_verification() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.onnx");
        fs::write(&path, b"abc").unwrap();
        let digest = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        assert_eq!(sha256_file(&path).unwrap(), digest);
        verify_sha256(&path, digest).unwrap();
        let error = verify_sha256(
            &path,
            "0000000000000000000000000000000000000000000000000000000000000000",
        )
        .unwrap_err();
        assert!(error.contains("SHA-256 mismatch"));
    }

    #[test]
    fn metadata_path_is_stable_for_the_source_identity() {
        let entry = Entry {
            file: "org--repo/onnx/model.onnx".into(),
            revision: Some("deadbeef".into()),
            sha256: None,
            heavy: false,
            coreml_unsupported: None,
            coreml_slow: None,
            override_dims: Default::default(),
            pin_inputs: Default::default(),
        };
        let root = Path::new("/cache");
        let first = metadata_path(root, &entry);
        assert_eq!(first, metadata_path(root, &entry));
        assert!(first.starts_with(root.join(METADATA_DIR)));
        assert_eq!(
            first.extension().and_then(|value| value.to_str()),
            Some("json")
        );
    }

    #[test]
    fn rejects_external_paths_that_escape_the_model_directory() {
        assert!(safe_relative_path("../weights.bin").is_err());
        assert!(safe_relative_path("/weights.bin").is_err());
        assert_eq!(
            safe_relative_path("weights/model.data").unwrap(),
            Path::new("weights/model.data")
        );
    }

    #[test]
    fn finds_unique_external_data_locations() {
        let tensor = TensorProto {
            name: "weight".into(),
            data_location: 1,
            external_data: vec![StringStringEntryProto {
                key: "location".into(),
                value: "model.onnx_data".into(),
            }],
            ..Default::default()
        };
        let model = ModelProto {
            graph: Some(GraphProto {
                initializer: vec![tensor.clone(), tensor],
                ..Default::default()
            }),
            ..Default::default()
        };
        assert_eq!(
            external_locations(&model).unwrap(),
            vec![PathBuf::from("model.onnx_data")]
        );
    }
}
