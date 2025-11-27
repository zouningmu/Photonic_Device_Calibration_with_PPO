clear; clc; close all;


if ~exist('testInputs','var')
    testInputs = rand(4,10);      
end

if ~exist('targetOutputs','var')
    targetOutputs = rand(4,10);   
end


env = MyCustomEnvPPO(testInputs, targetOutputs, @comput_output);

obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

obsDim = prod(obsInfo.Dimension);   % 16
actDim = prod(actInfo.Dimension);   % 15


%% ================================
%  Actor 网络: mean + variance
% ================================
actorLG = layerGraph();

% 输入
actorLG = addLayers(actorLG, featureInputLayer(obsDim,'Name','state'));

% 共享干路
actorLG = addLayers(actorLG, fullyConnectedLayer(64,'Name','fc1'));
actorLG = addLayers(actorLG, reluLayer('Name','relu1'));
actorLG = addLayers(actorLG, fullyConnectedLayer(64,'Name','fc2'));
actorLG = addLayers(actorLG, reluLayer('Name','relu2'));

% ===== Mean head =====
actorLG = addLayers(actorLG, fullyConnectedLayer(actDim,'Name','fc_mean'));
actorLG = addLayers(actorLG, tanhLayer('Name','mean_tanh'));
actorLG = addLayers(actorLG, scalingLayer('Scale',0.005,'Bias',0,'Name','mean'));

% ===== Variance head =====
actorLG = addLayers(actorLG, fullyConnectedLayer(actDim,'Name','fc_var'));
actorLG = addLayers(actorLG, reluLayer('Name','relu_var'));
actorLG = addLayers(actorLG, scalingLayer('Scale',0.1,'Bias',1,'Name','variance'));

% 连接 trunk
actorLG = connectLayers(actorLG,'state','fc1');
actorLG = connectLayers(actorLG,'fc1','relu1');
actorLG = connectLayers(actorLG,'relu1','fc2');
actorLG = connectLayers(actorLG,'fc2','relu2');

% 连接 mean head
actorLG = connectLayers(actorLG,'relu2','fc_mean');
actorLG = connectLayers(actorLG,'fc_mean','mean_tanh');
actorLG = connectLayers(actorLG,'mean_tanh','mean');

% 连接 variance head
actorLG = connectLayers(actorLG,'relu2','fc_var');
actorLG = connectLayers(actorLG,'fc_var','relu_var');
actorLG = connectLayers(actorLG,'relu_var','variance');

actorDlnet = dlnetwork(actorLG);
actorOpts = rlRepresentationOptions('LearnRate',3e-4);

actor = rlStochasticActorRepresentation( ...
    actorDlnet, ...
    obsInfo, ...
    actInfo, ...
    actorOpts );


%% ===========================================
%  Critic 网络
% ===========================================
criticLG = layerGraph();

criticLG = addLayers(criticLG, featureInputLayer(obsDim,'Name','state'));
criticLG = addLayers(criticLG, fullyConnectedLayer(64,'Name','c_fc1'));
criticLG = addLayers(criticLG, reluLayer('Name','c_relu1'));
criticLG = addLayers(criticLG, fullyConnectedLayer(64,'Name','c_fc2'));
criticLG = addLayers(criticLG, reluLayer('Name','c_relu2'));
criticLG = addLayers(criticLG, fullyConnectedLayer(1,'Name','value'));

criticLG = connectLayers(criticLG,'state','c_fc1');
criticLG = connectLayers(criticLG,'c_fc1','c_relu1');
criticLG = connectLayers(criticLG,'c_relu1','c_fc2');
criticLG = connectLayers(criticLG,'c_fc2','c_relu2');
criticLG = connectLayers(criticLG,'c_relu2','value');

criticDlnet = dlnetwork(criticLG);

criticOpts = rlRepresentationOptions('LearnRate',1e-3);

critic = rlValueRepresentation( ...
    criticDlnet, ...
    obsInfo, ...
    criticOpts );


%% ===========================================
%  PPO Agent
% ===========================================
agentOpts = rlPPOAgentOptions( ...
    'ClipFactor',0.2, ...
    'ExperienceHorizon',256, ...
    'MiniBatchSize',64, ...
    'DiscountFactor',0.99, ...
    'GAEFactor',0.95, ...
    'EntropyLossWeight',0.01);

%% ===========================================
%  Load or Create Agent
% ===========================================
modelFile = "trainedPPO_MZM_MZI.mat";

if isfile(modelFile)
    fprintf("检测到已有模型，加载中: %s\n", modelFile);
    tmp = load(modelFile,'agent');
    agent = tmp.agent;
else
    fprintf("未检测到已训练模型，新建 PPO agent...\n");
    agent = rlPPOAgent(actor, critic, agentOpts);
end

%% ===========================================
%  Training
% ===========================================
trainOpts = rlTrainingOptions( ...
    'MaxEpisodes',200, ...
    'MaxStepsPerEpisode',100, ...
    'Verbose',true, ...
    'Plots','training-progress');


trainingStats = train(agent, env, trainOpts);

save("trainedPPO_MZM_MZI.mat","agent");

bestPhase = env.MaxRewardPhase;
save("best_phase_current.mat","bestPhase");
