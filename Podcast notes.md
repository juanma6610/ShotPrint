Rex Omni 3 billion parameter multimodal model for object detection and visual perception next token/point predictor.

Unifies OCR, pointing,key point, pose estimation +more

Input: image+visual prompt 
Output: coordinates, 4 point polygons via discrete tokens instead of traditional regression.
Zero shot performance, matches even exceeds regression based detectors.

It seems to be open-source??
Github repo https://github.com/IDEA-Research/Rex-Omni

Doing a demo->Cant see but results seem very good??

Person detection works really well, it filters out crowd people and only focuses on players not on crowd or referee

It has a unified task formulation,
Instead of habing separate architectures for key pointing,OCR,etc it treats all of them as next point or sequence of coordinate token tasks->Really flexible
Quantized cooordinate tokenization, maps continuous values into discrete tokens [0,999] reduing learning difficulty of spatial outputs.
Dataset of 22million examples for supervised finetuning, then use RL post training to refine behaviors.
It handles arbitrary categories there are no preset ones.
Hnadles referencing objects by language, pointing t oobjects, visual prompting, key points for pose OCR and GUI grounding.

Inference time is prob heavy.

Could be useful to create preannotations for large datasets.

Could help in the creation of massive datasets,

Paper associated with the model https://arxiv.org/abs/2510.12798

---------------------------------------------------------------------------------------------------------------------------------------------------------------------


FINE QUEST ADAPTATIVE KNOWLEDGE ASSISTED SPORTS VIDEO UNDERSTANDING VIA AGENTS OF THOUGHTS REASONING

Paper: https://arxiv.org/abs/2509.11796

Ask question give 4 different answers model decides which is correct
Model uses reactive reasoining second model kicks in if 1st is indecisive it uses deliberative reasoning to answer more complex queries it has different modules and submodules, new contribution is the use of this dual model.

SOTA performance 50%-85% 
Problem is simplified since it is a multiple choice answer it doesnt come up with the answer.

Heavy on compute power. difficult to perform locally can have latency

------------------------------------------------------------------------------------------

Camera calibration-> link 2D positions of pixelsin the image space into the 3D real world space

https://arxiv.org/abs/2412.01721

Derive speed, know where players are etc

How to ensure model represent camera that does not move freely 

Use a tripod model, restrict in optimiztion movements of the camera by the possible movements of the tripod.

Intrinsic parameters change over time->lens,focus etc..

No need to calibrate camera with a chessboard