https://www.sloansportsconference.com/research-papers/characterization-of-space-and-time-dependence-of-3-point-shots-in-basketball



Intro and objective:
Analyze the relationship between **shot types** (Catch-and-Shoot vs. Pull-up) and **defensive pressure** using novel space-time models.
The authors introduce a **dynamic occupation model** that accounts for player **inertia (velocity and direction)**, rather than just static distance, to better quantify "openness"
- 0-2 feet very tight shot
- 2-4 feet tight shot
- 4-6 feet open shot
- >6 feet wide open shot

#### **Occupation Models (Quantifying "Openness")**

The paper proposes two ways to measure who controls a specific point on the court:

1. **Static Model ($\delta_{space}$):** Based purely on **distance**. It calculates the difference between the distance to the closest defender and the distance to the shooter.
    
2. **Dynamic Model ($\delta_{time}$):** Based on **time**. It calculates the difference between the time it takes for the defender to reach a point versus the shooter, accounting for their current speed and direction (inertia).
    
    - _Significance:_ If a defender is 5 feet away but moving in the wrong direction, the Static Model says they are "close," but the Dynamic Model correctly identifies that they will take a long time to contest the shot.
    ![[Pasted image 20251125194940.png]]
### **3. Key Findings**

#### Free space evolution before 3pt
Catch and shoot requires times to receive and shoot while pullups is only time to shoot. atch and shoot curves free space always decreases after shot in pullups it oscillates
#### **The "Inertia Lag" (Reaction Time)**

By comparing the Static and Dynamic models, the authors quantified the "delay" in defensive reaction.

- **finding:** There is a consistent **time lag of ~0.3 seconds** between the spatial opportunity appearing and the defender reacting.
    
- **Team Level:** On a global team level, the delay in adjusting defensive formation is significantly longer, around **1.26 seconds**.
    

#### **The "Zone of Death" for Shooters**

The study validated the critical thresholds for shot success.

- **Distance Threshold:** Accuracy drops drastically if a defender is within **6 feet** (traditional "open" definition).
    
- **Time Threshold:** The equivalent dynamic threshold is **0.4 seconds**. If a defender can reach the shooter in under 0.4s, shooting percentage plummets.
    
![[Pasted image 20251125195541.png]]

#### **Shooter Behavior & Tendencies**

The analysis revealed distinct archetypes of players based on how they handle space:

- **High-Volume/Elite Shooters (e.g., Curry, Korver):** They shoot regardless of pressure. They will take shots even if $\delta_{time}$ is low (closely guarded).
    
- **Conservative Shooters (e.g., Porter, Russell):** They only shoot if $\delta_{time} > 0.5s$ (wide open). If the defender is closer, they pass up the shot.
    
- **Release Time:** Elite shooters (success rate >40%) consistently have a **faster-than-average release time**, suggesting that the ability to shoot quickly is a prerequisite for high-level efficiency.

Some shooters adapt their shots based on the space they have taking longer to shoot others stay consistent in release time.