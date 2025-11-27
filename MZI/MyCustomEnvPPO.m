classdef MyCustomEnvPPO < rl.env.MATLABEnvironment
    properties
      
        PhaseCurrent           

        
        Observation             

        
        MaxRewardValue = -Inf
        MaxRewardPhase
        
        
       
        TestInputs             

        
        TargetOutputs           

       
        ChipFunction = @comput_output
    end
    
    methods
        function this = MyCustomEnvPPO(testInputs, targetOutputs, chipFcn)
           
            if nargin < 1
                error('必须传入 testInputs 和 targetOutputs');
            end
            if nargin < 2
                error('必须传入 targetOutputs');
            end
            if nargin < 3
                chipFcn = @comput_output;  
            end

          
            
           
            ObservationInfo = rlNumericSpec([16 1], ...
                'LowerLimit', -inf, ...
                'UpperLimit', inf);
            ObservationInfo.Name = 'State';
            ObservationInfo.Description = 'Phase current (15) + current loss (1)';

           
            ActionInfo = rlNumericSpec([15 1], ...
                'LowerLimit', 0, ...
                'UpperLimit', 1);
            ActionInfo.Name = 'PhaseCurrentAction';
           
            this = this@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);
            this.TestInputs   = double(testInputs);
            this.TargetOutputs = double(targetOutputs);
            this.ChipFunction  = chipFcn;

            
            this.PhaseCurrent = 0.01 * rand(15,1);

           
            loss = evaluateLoss(this, this.PhaseCurrent);

           
            this.Observation   = [this.PhaseCurrent; loss];
            this.MaxRewardPhase = this.PhaseCurrent;
        end

       
        function [Observation,Reward,IsDone,LoggedSignals] = step(this,Action)

           
            this.PhaseCurrent = double(Action(:));

            
            loss = evaluateLoss(this, this.PhaseCurrent);

            
            Reward = 0.01 - loss;

            
            if ~isfinite(Reward) || ~isreal(Reward)
                Reward = -1e3;
            end

            
            if Reward > this.MaxRewardValue
                this.MaxRewardValue = Reward;
                this.MaxRewardPhase = this.PhaseCurrent;
            end

            Observation = [this.PhaseCurrent; loss];
            this.Observation = Observation;

          
            IsDone = false;

            LoggedSignals = [];
        end

    
        function InitialObservation = reset(this)
            
            this.PhaseCurrent = 0.01 * rand(15,1);

          
            loss = evaluateLoss(this, this.PhaseCurrent);

           
            InitialObservation = [this.PhaseCurrent; loss];
            this.Observation   = InitialObservation;
        end
    end

    methods (Access = private)
        function loss = evaluateLoss(this, phase_15)
            

            phase_15 = double(phase_15(:));
            
            phase_15 = max(0,min(0.01,phase_15));

            [d, numCases] = size(this.TestInputs);
            se_sum = 0;

            for k = 1:numCases
                x = this.TestInputs(:,k);        
                y_target = this.TargetOutputs(:,k); 

                
                y = this.ChipFunction(x, phase_15); 

                e = y - y_target;
                se_k = sum(e.^2);
                se_sum = se_sum + se_k;
            end

            loss = se_sum;

            
            if ~isfinite(loss) || ~isreal(loss)
                loss = 1e3;
            end
        end
    end
end
