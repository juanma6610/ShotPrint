https://arxiv.org/pdf/1408.0777

## Summary
Paper proposes Expected Possesion Value (EPV) metric that estimates number of points an offense will score at the end of the possesion given the full spatiotemporal configuration of players and ball at any moment in time.
Traditional "box score" metrics (ppg, assists, field goals) are focused in discrete  make or miss metrics, EPV is a continuous, real time validation of decision-making it acts as a "stock ticker" for a possesion. ![[Pasted image 20251125145639.png]]

## Methodological framework
Multiresolution stochastical process designed to handle the high dimensionality of optical tracking data (25Hz, X/Y coordinates for 10 players + ball).
The game is decomposed into two levels of resolution.
**1.Macro-transition (Coarsened process) **
Model views possesion as a sequence of discrete events a Markov chain approach->macrotransitions (major events that decouple the future of the possession from its immediate history, specifically: passes, shot attempts, and turnovers.)
![[Pasted image 20251125150212.png]]

![[Pasted image 20251125150410.png]]
C$_{end}$ = { made 2pt, made 3pt, end of possession} 
C$_{trans}$ = {shot attempt from C$_{poss}$ , pass to c $_{poss}$ from c $_{poss}$, turnover in progress, rebound in progress} 
These transition states carry information about the possession path, such as the most recent ballcarrier, and the target of the pass, while the ball is in the air during shot attempts and passes

For simplicity and limitations excludes several notable basketball events (such as fouls, violations, and other stoppages in play) and aggregates others (the data, for example, does not discriminate among steals, intercepted passes, or lost balls out of bounds).

**Competing Risks Model:** The transition from one state to another is modeled using a **competing risks framework**. At any instant, a ball-handler is "at risk" of passing, shooting, or turning it over. These hazards are modeled log-linearly based on spatial and situational covariates.
This is done in practice with a transition probability matrix
![[Pasted image 20251125151314.png]]

**2.The Micro-Transition (Fine-Grained Process) **

Describes player movement with the ball-carrier held constant, between macrotransitions

 It utilizes a Taylor series expansion where a player's future location depends on their current velocity (first derivative) plus a stochastic innovation term (representing acceleration/jerk).

**Fields:** This allows the creation of "velocity fields" or "acceleration fields" for specific players, visualizing where a player like Tony Parker or Dwight Howard tends to move given a specific court configuration.
![[Pasted image 20251125151951.png]]
There is a different model for offense and defense

Creation of a adjacency matrix to model player similarity

Inference and estimation
### 1. The Hierarchical Model Structure

To ensure robust estimates—even for players or court zones with sparse data—the authors use a **hierarchical model**. This allows "information sharing" across the league.

- **The Problem:** Estimating a unique transition probability for every player at every specific location on the court is impossible because no player visits every spot enough times.
    
- **The Solution:** The model decomposes the hazard function (the instantaneous probability of an event) into **shared** and **player-specific** components.
    
    - **Global Effects:** Spatial patterns (e.g., "shooting from right under the basket is generally high probability") are learned from _all_ players.
        
    - **Player Deviations:** Individual player parameters capture how a specific player (e.g., LeBron James) deviates from the league average.
        

### 2. Poisson Regression for Hazards

transform the problem of estimating transition rates into a **Poisson regression**.

- **Discretization:** The continuous time of a possession is broken into tiny intervals. If the intervals are small enough, the likelihood of an event occurring (passing, shooting) is proportional to the hazard rate.
    
- Log-Linear Model: The log-hazard for a player $\ell$ performing action $j$ (e.g., passing) at time $t$ is modeled as:
    
    $$\log(\lambda_{\ell j}(t)) = \text{Situational Covariates} + \text{Spatial Effects} + \text{Player-Specific Offsets}$$
    
    - **Situational Covariates:** Things like dribble duration or distance to the closest defender.
        
    - **Spatial Effects:** Represented by basis functions ($\phi(z)$) that map the court surface.
        

### 3. Computational Strategy: INLA vs. MCMC

The most distinct technical choice in this section is the rejection of Markov Chain Monte Carlo (MCMC) in favor of **INLA** due to the sheer scale of the data.

- **The Data Scale:** The design matrix for these regressions contains approximately **30.4 million rows** and over **6,000 columns**.
    
- **Why INLA?:** Standard MCMC would take too long to converge on a dataset this size. INLA provides a fast, deterministic alternative for Bayesian inference by approximating the posterior distributions using Laplace approximations.
    
- **Implementation:** Even with INLA, the computation is massive. They fit the models by splitting the task across **461 processors**, with each macrotransition type requiring up to 16 hours to fit.
    

### 4. Spatial Basis Functions

To handle the "where" of the events, the inference section relies on a spatial discretization of the court.

- **Finite Element Method:** They discretize the court into a triangular mesh.
    
- **Basis Functions:** They construct piecewise linear basis functions over this mesh. This allows the model to learn smooth spatial surfaces (e.g., a "shot map") without assuming a rigid parametric form (like a 2D Gaussian).

|**Component**|**Method Used**|**Purpose**|
|---|---|---|
|**Model Type**|Hierarchical Poisson Regression|To model event hazards (rates) over time.|
|**Regularization**|Bayesian Hierarchical Priors|To share data between players and prevent overfitting on rare events.|
|**Inference Engine**|**INLA** (R-INLA package)|To approximate posteriors feasibly on 30M+ rows of data.|
|**Spatial Inputs**|Triangular Mesh Basis Functions|To capture complex, non-linear spatial tendencies on the court.|


New metrics added:
- EPVA: quantifes a players overall offensive value through his movements and decisions while handling the ball, relative to the estimated value contributed by a league-average player receiving ball possession in the same situations.
![[Pasted image 20251125154536.png]]can be related to developing players why are they most succesful in this situation??(height advantage, skills, driving ability etc)
- Shot satisfaction: For each shot attempt a player takes, we wonder how satisfied the player is with his decision to shoot what was the expected point value of his most reasonable passing option at the time of the shot?

Assumptions made:
- View of bball possesion is semi-Markov->violates pre set plays by coach
- Model rebounds based on player profiles and dynamics of movements before shots (now it is a 50-50prob to go either way)
- No distiction between types of turnovers (steals, bad passes, ball out of bounds)
- The data that feeds the model lacks important vars such as positioning of player hands and feet, height when jumping etc
- Does the data being fed have any link with each players ability??? (shooting % drive abilitities etc...)