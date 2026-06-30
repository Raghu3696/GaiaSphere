# GaiaSphere Architecture

## Design Philosophy

GaiaSphere treats the Earth's surface as it is: a **dynamic, multi-scale, spatially connected system**. The architecture mirrors that structure:

- **ViT-Geo** sees each satellite frame as a set of spatial tokens — no assumptions about which pixels matter
- **ST-Former** understands how those tokens change over days/weeks — capturing flood onset, crop stress progression, fire growth
- **GeoGraph** knows that pixels aren't isolated — what happens upstream affects downstream, what burns upwind threatens downwind

## Data Flow

```
                     ┌───────────────────────────────────┐
                     │           Input Modalities         │
                     └───────────────────────────────────┘
                           │              │           │
                    Sentinel-2        ERA5 Weather   SRTM DEM
                   [B,T,13,H,W]       [B,T,16]     [B,1,H,W]
                           │              │           │
                           ▼              │           ▼
                       ┌───────┐          │      ┌──────────┐
                       │ViT-Geo│          │      │DEMEncoder│
                       │       │          │      │   (CNN)  │
                       └───────┘          │      └──────────┘
                           │              │           │
                    [B,T,N,768]      [B,T,768]   [B,N,256]
                           │              │
                           ▼              ▼
                       ┌──────────────────────┐
                       │      ST-Former        │
                       │  (Factorized Attn)    │
                       └──────────────────────┘
                                  │
                            [B,N,768]
                                  │
                           ┌──────┴──────┐
                           │   Fusion    │   ← concat + project
                           │  (MLP)      │
                           └──────┬──────┘
                            [B,N,256]
                                  │
                           ┌──────▼──────┐
                           │  GeoGraph   │   ← graph propagation
                           │   (GAT+SAGE)│
                           └──────┬──────┘
                            [B,N,256]
                           ┌──────┼──────┐
                           ▼      ▼      ▼
                        Flood   Crop  Wildfire
                         Head   Head    Head
```

## Why These Choices?

| Decision | Rationale |
|----------|-----------|
| ViT over CNN for imagery | Global attention captures long-range spatial correlations (e.g., a river's watershed) that convolutions miss |
| Factorized ST attention | Full space-time attention is O(T²N²) — factorization makes it O(TN² + T²N), tractable for month-long windows |
| GNN over spatial CNN for propagation | Watersheds, parcels, and fire risk zones are irregular — graphs handle arbitrary topology naturally |
| Focal loss for wildfire | Fire pixels are ~1-5% of any tile — standard BCE collapses to predicting all-negative |
| Geo-aware positional encoding | A patch at 23°N behaves differently than one at 60°N (vegetation, climate, sun angle) — the model should know |
