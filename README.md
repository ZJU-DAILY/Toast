# Toast
Python implementation of Toast: Task-oriented Augmentation for Spatio-Temporal Data.

## Dependencies
- Pytorch 2.1.2
- pytorch_geometric 2.5.3
- traj_dist 1.1
- numpy 1.26.3

## TODO
- [x] Build an Encoder-Decoder architecture
- [x] Prepare necessary parts for model training
- [x] Downstream task 1 (trajectory recovery)
- [ ] Design Augmentor
  - [ ] implement GPS point union
  - [ ] implement trajectory union
  - [ ] implement attribution join
- [ ] Task-oriented augmentation
- [ ] Downstream task 2 (trajectory similarity searching)
- [ ] Downstream task 3 (TBD)