The current implementation is for rolling-contact-qp.tex. However, there was an mistake, namely, each steering wheel merely include a single rolling velocity(about the pitch), without considering the steering velocity (about the yaw). Hence, in the mc_mujoco simulation, the (a four-steering wheel) robot can not correctly response to the key board rotation command, QE, and linear velocity command WASD. 

Now, there is a corrected version, rolling-contact-qp-new.tex, which should have fixed the theory. 

Review git diff, and recent commit e21ed9a997ed73bafbb345a82f92bcdfb1c28585, and create  a work plan for implementing the corrections in  rolling-contact-qp-new.tex, especially the four steering wheel QP (\eqref{eq:four-steering-wheel-qp}) for the Ranger Mini V3 robot. 

In the end, by running: "/home/yuquan/.local/bin/mc_rtc_ticker -s robot:=RollingContactRangerMiniV3", we should drive around the simulated robot smoothly. 
