https://dspace.mit.edu/bitstream/handle/1721.1/139205/Jutamulia-ivanj-meng-eecs-2021-thesis.pdf?sequence=1&isAllowed=y


Decisions should not be characterized as good or bad based on the actual outcome, but rather the expectation of the outcome.
Decision making evaluation should be rooted in expectation.
Expected Possession Value (EPV) which seeks to characterize the expected number of points for every player during an offensive possession at any point in time. EPV captures the hypothetical scenario where the ball is passed to a particular player and they take a shot, quantifying how much value that particular event would contribute to the team’s point total.
![[Pasted image 20251125172917.png]]
![[Pasted image 20251125173117.png]]
![[Pasted image 20251125173204.png]]


- **Core Problem:** Evaluating decisions based on outcomes (e.g., a made shot) is flawed. A good decision (passing to a wide-open teammate) remains "good" even if the teammate misses the shot.
    
- **Proposed Solution:** The author introduces **Expected Possession Value (EPV)**, a framework rooted in expectation rather than result. It quantifies the value of every possible decision (shoot or pass) at any moment in a possession.
    

### **2. Project Overview & Components**

The research was conducted in collaboration with the **San Antonio Spurs** and utilized **Google Cloud** for computing power.  Second Spectrum data accumulated from the 2013-2014 NBA season to the 2018-2019 NBA season, amounting to 8,210 games, 1.5 million unique possessions, and nearly 80,000 hours of tracked game-time.

### **3. Methodology: Difficulty Models**

To calculate EPV, the system must first answer two fundamental questions for the ballhandler: _1) Is this a good shot?_  2) Is there a pass that leads to a better shot?. This requires quantifying difficulty:

#### **Pass Difficulty Model**

- **Goal:** Calculate the probability of a pass being successfully completed.
    
- **Approach:**  tested both **Logistic Regression** and **Neural Network** approaches. ML algorithms are supervised.

- Dataset: 786,208 passes from the 2018-2019 NBA season as data points, splitting them into training and evaluation sets 70% to 30%. data is biased towards positive examples. More specifically, there is a disproportionately high number of completed passes in the dataset due to players opting not to make difficult passes.
    
- **Features:** Geometric features extracted from tracking data, such as the angle of the pass, the distance to the nearest "obstructive" defender, and the configuration of the defense. ![[Pasted image 20251125175424.png]]
    ![[Pasted image 20251125175732.png]]
- Neural network approach is really simple only one hidden layer with 128 neurons
- Neural network outperforms logistic regression but lacks interpretability
    
- **Output:** A probability between 0 and 1 indicating likelihood of completion.
    

#### **Shot Difficulty Model**

- **Goal:** Calculate the probability of a shot going in (making the basket).
    
- **Features:** Similar to passing, this uses geometric data like the shooter's distance to the hoop, the defender's contest angle ($alpha$), and the shooter's movement vectors (fading left/right).![[Pasted image 20251125182816.png]]
    
- **Architecture:** A Neural Network was used to handle the complex non-linear relationships in shot dynamics, deeper than the pass neural network, 4 hidden layers each with 512 neurons.
    

### **4. Expected Possession Value (EPV)**

#### **Definition**

EPV is defined as the **expected number of points** to be scored by an offensive player if they receive the ball and shoot at that specific instance in time.

#### **Calculation**

At every time frame (25 frames/sec), the system calculates 5 values for the ballhandler:

1. **Value of Shooting:** (Probability of making the shot) $\times$ (Points for the shot).
    
2. Value of Passing (to each of 4 teammates): (Probability of completing pass) $\times$ (Teammate's probability of making shot) $\times$ (Points).
    
    Note: This simplifies the game into binary shoot/pass decisions to isolate specific decision values.
    

#### **Visualization**

EPV as an evolving metric throughout a possession.

- **Dot Diagrams:** Visuals where the size of a player's dot correlates to their current EPV (larger dot = better scoring opportunity)19.
    
- **Evolution Plots:** Line graphs tracking the EPV of all 5 players over the shot clock, allowing analysts to spot "spikes" (missed opportunities) or "valleys" (poor spacing).
    ![[Pasted image 20251125183835.png]]
    

### **5. Evaluation of the Metric**

To validate EPV, compare the **aggregated EPV predictions** against **actual game scores** for games in January 2019.

- **Accuracy:** On average, EPV predicts the final score correctly (error centered around 0)22.
    
- **Bias to Mean:** The model exhibits a "bias towards the mean." It underpredicts scores for high-performing teams (like the Golden State Warriors) and overpredicts for poor-performing teams.
    
    - _Reasoning:_ EPV assumes "average" player ability. It doesn't inherently know that Steph Curry is a better shooter than an average player.
        
- **Correction:** The author improved accuracy by applying a correction factor based on a team's **Field Goal Percentage (FG%)**, significantly reducing prediction error.
    

### **6. Application: Missed Opportunities**



- **Defining Opportunities:** An "opportunity" is defined as a moment where a player's EPV exceeds a certain threshold (e.g., >1.5 expected points. for 2 seconds or longer
    
- **Missed Opportunity:** A high-EPV window where the ball was _not_ passed to the open player.
    
- **Metrics:**
    
    - **Strategy:** Measured by the _number_ of opportunities generated. High count = good schema.
        
    - **Execution:** Measured by the _missed opportunity rate_. High rate = players failing to find the open man.
    The "Cost" of Hesitation (Missed Opportunity Delta)
    Good Execution, "Bad" Strategy
    Team Strategy vs. Execution

| **Metric**                  | **What it Measures**        | **High Value Meaning**                          | **Low Value Meaning**                                  |
| --------------------------- | --------------------------- | ----------------------------------------------- | ------------------------------------------------------ |
| **Number of Opportunities** | **Strategy (The Plan)**     | The playbook successfully confuses the defense. | The offense is stagnant; relying on isolation scoring. |
| **Missed Opportunity Rate** | **Execution (The Ability)** | Players are hesitating or missing reads.        | Players are decisive and finding the open man.         |

    

### **7. Discussion & Limitations**

- **Quantifying "Points Left on the Table":** The analysis can estimate how many additional points a team _could_ have scored if they capitalized on all generated opportunities31.
    
- **Limitations:**
	- Shooting ability of players->Can be improved by putting individual metrics or categorizing shooters.
	- Defensive ability of defenders is also not taken into account
	- Does not take into account fouls while shooting
    
    - **2-for-1 Situations:** EPV might classify a rushed shot at the end of a quarter as a "bad decision" (low expectation). However, strategically, taking a quick shot to ensure a final possession is a "good decision" that the current EPV model does not capture.
        
    - **Oversimplification:** The model focuses purely on immediate pass/shoot probabilities and does not account for complex future actions (e.g., passing to a player so _they_ can drive and kick) .
        

### **8. Conclusion**

The thesis concludes that EPV is a valid, powerful framework for separating process from result. It allows teams to objectively evaluate decision-making, identify specific breakdowns in execution, and quantify the effectiveness of their offensive strategies beyond simple point totals.