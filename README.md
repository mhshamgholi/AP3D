#### [experiments excel](https://docs.google.com/spreadsheets/d/1fVqYhwlGdbjq8LHpuDbNoWAIeYPBh96jnMIYqy773DQ/edit?usp=sharing)

#### Requirements: 

Python=3.6 and Pytorch=1.0.0

#### Training and test

  ```Shell
  # For MARS
  python train.py --root /home/guxinqian/data/ -d mars --arch ap3dres50 --gpu 0,1 --save_dir log-mars-ap3d #
  python test-all.py --root /home/guxinqian/data/ -d mars --arch ap3dres50 --gpu 0 --resume log-mars-ap3d
  
  ```
