clear; clc; close all;

%% ================================
%  Create Environment
% ================================
env = MyCustomEnvPPO();
obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

obsDim = prod(obsInfo.Dimension);
actDim = prod(actInfo.Dimension);


%% ================================
%  Correct Actor for OLD PPO API
% ================================
actorLG = layerGraph();

% Input
actorLG = addLayers(actorLG, featureInputLayer(obsDim,'Name','state'));

% Shared layers
actorLG = addLayers(actorLG, fullyConnectedLayer(64,'Name','fc1'));
actorLG = addLayers(actorLG, reluLayer('Name','relu1'));
actorLG = addLayers(actorLG, fullyConnectedLayer(64,'Name','fc2'));
actorLG = addLayers(actorLG, reluLayer('Name','relu2'));

% ===== Mean head =====
actorLG = addLayers(actorLG, fullyConnectedLayer(actDim,'Name','fc_mean'));
actorLG = addLayers(actorLG, tanhLayer('Name','mean_tanh'));
actorLG = addLayers(actorLG, scalingLayer('Scale',0.005,'Bias',0,'Name','mean')); 
% 最终输出层名 = mean

% ===== Variance head =====
actorLG = addLayers(actorLG, fullyConnectedLayer(actDim,'Name','fc_var'));
actorLG = addLayers(actorLG, reluLayer('Name','relu_var'));  
actorLG = addLayers(actorLG, scalingLayer('Scale',0.1,'Bias',1,'Name','variance'));
% 最终输出层名 = variance  (必须叫这个)

% ===== Connect trunk =====
actorLG = connectLayers(actorLG,'state','fc1');
actorLG = connectLayers(actorLG,'fc1','relu1');
actorLG = connectLayers(actorLG,'relu1','fc2');
actorLG = connectLayers(actorLG,'fc2','relu2');

% ===== Connect mean head =====
actorLG = connectLayers(actorLG,'relu2','fc_mean');
actorLG = connectLayers(actorLG,'fc_mean','mean_tanh');
actorLG = connectLayers(actorLG,'mean_tanh','mean');

% ===== Connect variance head =====
actorLG = connectLayers(actorLG,'relu2','fc_var');
actorLG = connectLayers(actorLG,'fc_var','relu_var');
actorLG = connectLayers(actorLG,'relu_var','variance');

% Convert to dlnetwork
actorDlnet = dlnetwork(actorLG);

% Representation options
actorOpts = rlRepresentationOptions('LearnRate',3e-4);


actor = rlStochasticActorRepresentation( ...
    actorDlnet, ...
    obsInfo, ...
    actInfo, ...
    actorOpts );


%% ===========================================
%  Critic Network
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
modelFile = "trainedPPO_MZM.mat";

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

fprintf("开始训练 PPO...\n");
trainingStats = train(agent, env, trainOpts);

save("trainedPPO_MZM.mat","agent");
piphasecurrent = env.MaxRewardInput;
save("pi_current","piphasecurrent");
