# ROS2 Integrated Geiger Counter
## What is this?
This package is part of a project developing long-term surveillance drones for nuclear site monitoring. One of the sensors onboard the robots is a small-form factor and low-cost Geiger-Muller (GM) counter connected to the embedded computer of the robot. 
## What does it do?
This package does the following: 
1. Reads the raw data throughput every time the sensor "clicks" (reads a count). 
2. Calculates (a) counts per second, (b) counts per minute, (c) millirems per hour. 
3. Publishes the counts per minute and millirems per hour on a topic called /rad. 
## What can I do with this data? 
Anything, really. 
