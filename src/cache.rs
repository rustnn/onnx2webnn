/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Platform-appropriate cache locations shared by the CLI and model validator.

use std::path::PathBuf;

const APPLICATION_DIR: &str = "onnx2webnn";

/// Return the directory for cached source ONNX models.
///
/// `O2W_ONNX_CACHE` overrides only this cache. `O2W_CACHE_DIR` relocates all
/// onnx2webnn caches while retaining their `onnx`/`webnn` subdirectories.
pub fn onnx_cache_dir() -> Result<PathBuf, String> {
    cache_dir("O2W_ONNX_CACHE", "onnx")
}

/// Return the directory for exported WebNN graphs and Safetensors archives.
///
/// `O2W_WEBNN_CACHE` overrides only this cache. `O2W_CACHE_DIR` relocates all
/// onnx2webnn caches while retaining their `onnx`/`webnn` subdirectories.
pub fn webnn_cache_dir() -> Result<PathBuf, String> {
    cache_dir("O2W_WEBNN_CACHE", "webnn")
}

fn cache_dir(specific_env: &str, kind: &str) -> Result<PathBuf, String> {
    resolve_cache_dir(
        nonempty_env(specific_env),
        nonempty_env("O2W_CACHE_DIR"),
        dirs::cache_dir(),
        kind,
    )
    .ok_or_else(|| {
        format!(
            "cannot determine the operating-system cache directory; set {specific_env} or O2W_CACHE_DIR"
        )
    })
}

fn nonempty_env(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
}

fn resolve_cache_dir(
    specific: Option<PathBuf>,
    shared: Option<PathBuf>,
    os_cache: Option<PathBuf>,
    kind: &str,
) -> Option<PathBuf> {
    specific.or_else(|| {
        shared
            .or_else(|| os_cache.map(|root| root.join(APPLICATION_DIR)))
            .map(|root| root.join(kind))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn cache_path_precedence_is_specific_shared_then_os() {
        let specific = PathBuf::from("/specific");
        let shared = PathBuf::from("/shared");
        let os = PathBuf::from("/os-cache");

        assert_eq!(
            resolve_cache_dir(
                Some(specific.clone()),
                Some(shared.clone()),
                Some(os.clone()),
                "onnx"
            ),
            Some(specific)
        );
        assert_eq!(
            resolve_cache_dir(None, Some(shared), Some(os.clone()), "onnx"),
            Some(Path::new("/shared/onnx").to_path_buf())
        );
        assert_eq!(
            resolve_cache_dir(None, None, Some(os), "webnn"),
            Some(Path::new("/os-cache/onnx2webnn/webnn").to_path_buf())
        );
        assert_eq!(resolve_cache_dir(None, None, None, "onnx"), None);
    }
}
