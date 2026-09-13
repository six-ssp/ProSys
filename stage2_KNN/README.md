# Stage 2 Compatibility Shim

The maintained parallel KNN + ReaFNN Stage-2 implementation is in
[`../stage2_ReaFNN/`](../stage2_ReaFNN/). This package intentionally re-exports
the public classes under the historical `stage2_KNN` import path so that older
launchers and external scripts remain runnable. Do not add new Stage-2 logic
here.

See [`../stage2_ReaFNN/stage2_ReaFNN_detail.md`](../stage2_ReaFNN/stage2_ReaFNN_detail.md)
for the implementation and protocol.
