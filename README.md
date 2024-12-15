Setup the environment

```
conda create -n balance-bot python==3.8
conda activate balance-bot
pip install -e . --config-settings editable_mode=compat
```

Run the PID balance code
```
cd ./tests
python pid_agent.py
```

Run the keyboard navigation code
```
cd ./tests
python keyboard_nav_agent.py
```

Run the PID balance integrated with RL navigation code
```
cd ./tests
python pid_navigation_agent.py
```

Test the trained model using the code:
```
cd ./tests
python evaluate.py
```

change the parameters such as Kp, Kd and Ki, toggle between terrains, camera position and target position in ```helper/config.py``` file