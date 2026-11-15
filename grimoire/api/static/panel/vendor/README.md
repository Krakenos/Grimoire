# Vendored frontend libraries

Third-party JavaScript used by the management panel is committed here rather than
loaded from a CDN, so the panel works in fully offline / air-gapped deployments
(no outbound network needed to render the page).

Each file is pinned to an exact version and verified by SHA-256. To refresh or
re-verify, run `./fetch.sh`.

## vis-network

| | |
|---|---|
| **File** | `vis-network.min.js` |
| **Version** | 10.1.0 |
| **Build** | standalone UMD (bundles `vis-data`; exposes the global `vis`) |
| **Upstream** | https://github.com/visjs/vis-network |
| **Source URL** | https://unpkg.com/vis-network@10.1.0/standalone/umd/vis-network.min.js |
| **License** | Apache-2.0 OR MIT (dual-licensed; header retained in the file) |
| **SHA-256** | `fd730e304a5b877a937a896be9536e7974dc473d8ac87fa66644bce52cb5f8e4` |

Used by `app.js` for the memory-graph tab (`vis.DataSet`, `vis.Network`).
The **standalone** build is required — the plain `dist/` build does not include
`vis-data`, so `vis.DataSet` would be undefined.
