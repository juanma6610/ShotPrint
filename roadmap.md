✅ What is Already Completed
1. Base Game Physics & Tracking (game.py)
Foundational methods to extract the raw spatial data. This includes:Player positions, velocity, and acceleration per frame.Shot detection (made/missed times).Advanced spatial metrics like time-to-reach, Voronoi areas, space control heatmaps, and convex hulls (team spacing).Contextual data like offensive possession and shot clock remaining. Also created a small UI to be able to visualize plays and frames of the plays.
2. Phase 1: Defender Archetype Clustering 
We successfully completed the defensive half of Phase 1.We scraped, cleaned, and merged the defensive tracking data from the NBA API.We applied a >20 games played filter to remove noise.We ran K-Means clustering to create 5 distinct Defensive Archetypes (Elite Rim Protectors, High-Volume Perimeter Chasers, Versatile Rotation Defenders, Defensive Liabilities, and Drop-Coverage Bigs).We exported this mapping to defender_archetypes_15_16.csv ready for One-Hot Encoding. Used also PCA and tried gaussian mixture models as well.
3. Shooter Archetype Clustering: scraped data from basketball reference
cluster the offensive players using K-Means, build clusters based on:Shooting ability profile (FG% by zone).Range preference (Average shot distance).Shot volume (Attempts per minute).Catch-and-shoot percentage and speed at shot time (Stationary vs. off-movement). ALso used PCA and tried gaussian mixture models as well.
Here are the archetypes:Explanation of the clusters obtained:
    1. The Spot-Up Spacers / 3&D Wings (Cluster 1, n=110)
    The Profile: Trevor Ariza, Kentavious Caldwell-Pope, Wesley Matthews, P.J. Tucker.
    2. Primary Creators / Ball-Dominant Guards (Cluster 2, n=82)
    The Profile: James Harden, Kemba Walker, Kyle Lowry, Paul George.
    3. Mid range and all around shooters (Cluster 3, n=67)
    The Profile: DeMar DeRozan, Khris Middleton, Klay.
    4. Traditional Paint Anchors / Lob Threats (Cluster 4, n=46)
    The Profile: DeAndre Jordan, Andre Drummond, Dwight Howard.
    5. Athletic Slashers / Power Forwards (Cluster 5, n=44)
    The Profile: Giannis Antetokounmpo, Jabari Parker, Julius Randle.

Work in progress
2. Phase 2: Feature Extraction (shot_features.py)
We need to generate the actual training matrix for the Neural Network. For every detected shot, we must calculate:The 16 Baseline Features: Shot distance, shot angle, closest defender distance/angle, and parallel/perpendicular velocities and accelerations for both the shooter and closest defender.Your 10 Novel Features: Physics-based time-to-contest, offensive/defensive convex hulls, space control delta-time, 2nd closest defender pressure, and the One-Hot Encoded shooter/defender archetypes.

3. Phase 3: Neural Network Training (shot_model.py)
Once the features are built, we will transition fully into PyTorch to build the model:Construct a 4-layer feed-forward network with Dropout and a Sigmoid output layer for binary classification.Train the model using a 70/15/15 (train/val/test) split to predict the binary target (made_shot).The primary goal is to beat the prior baseline accuracy of 63%.

4. Verification & Evaluation
Finally, we will validate the model:Run Feature Importance / SHAP analysis to prove that your novel features (like time_to_contest and our newly built Archetypes) actually improve prediction.Generate a calibration plot to ensure predicted probabilities match real make rates.Conduct an ablation study (testing the model with and without our archetypes/physics features). 