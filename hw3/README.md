# environment
安装miniconda（或anaconda） 可参照https://www.anaconda.com/docs/getting-started/miniconda/main

```bash
cd hw3
conda create -n kalman_sim python=3.11
conda activate kalman_sim
pip install -r requirements.txt
```

# hw3
TODO in main.py

调整ekf参数，使得在不同的目标运动场景下能有效地跟踪（稳定跟随和快速跟随的tradeoff）