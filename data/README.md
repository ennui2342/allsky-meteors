`nsucp_profiles.npy` — 4054 x 64 float32: brightness along the trail of real
meteors (peak 1, resampled to 64 points, direction of flight unknown),
extracted by `../tools/nsucp_eval.py` from the sum images of the NightSkyUCP
dataset (Sennlaub, Hofmann et al., MNRAS 2022, "Object classification on
video data of meteors and meteor-like phenomena"; TU Ilmenau, AMS / AllSky7),
doi:10.6084/m9.figshare.16451625, licensed CC BY-NC-SA 4.0 (this derived
file carries the same licence). Used by `scorer inject --profiles`.

`labels.csv` — the user's verdicts on scorer candidates, keyed by frame time
(UTC, end of exposure) and streak end points in saved-frame pixels: `good`
= a meteor worth showing; `not_good` = not a meteor, or not worth showing.
The goal is precision on good-looking meteors, not a full count.
