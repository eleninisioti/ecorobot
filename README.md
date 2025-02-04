# Ecorobot: evaluating complex locomotion and navigation

Ecorobot is a benchmark that enables the evaluation of agents in tasks requiring complex locomotion and navigation.
Built on top the Brax physics-engine, it offers an interface for designing tasks where robots of customizable morphologies equipped with different types of sensors interact with items such as walls, obstacles and food.
In addition to enabling the design of custom tasks, we have provided a set of tasks largely inspired from previous NE studies that test for specific behavioral skills.

![Alt Text](images/envs.gif)

An ecorobot environment is a combination of a robotic morphology (that can be equipped with sensors) and a task

## Robots
We currently support some of the Brax environmets (the ones that can locomote) and two simple custom robots:

* ant
* halfcheetah
* swimmer
* walker2d
* hopper
* simple-rob
* simple-rob-turn

## Tasks

Tasks are formed by positioning modules (such as food and walls) and choosing a reward function. 
Currently the following tasks are available:

* locomotion 
* deceptive-maze-easy
* deceptive-maze-difficult
* maze-with-stepping-stones
* hierarchical obstacles


## How to run

To install all package requirements you can run

```
pip install requirements.txt
```

For examples of how to run the different tasks see [this examples script](scripts/play.py)