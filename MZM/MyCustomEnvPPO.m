classdef MyCustomEnvPPO < rl.env.MATLABEnvironment
    properties
        
        CurrentInput        
        Observation
        MaxRewardValue = -Inf
        MaxRewardInput
        TargetFunction = @compute_input %Outside interface
    end
    
    methods
        function this = MyCustomEnvPPO()
            ObservationInfo = rlNumericSpec([5 1], ...
                'LowerLimit', -inf, ...
                'UpperLimit', inf);
            ObservationInfo.Name = 'State';

          
            ActionInfo = rlNumericSpec([4 1], ...
                'LowerLimit', 0, ...
                'UpperLimit', 1);
            ActionInfo.Name = 'Action';

            this = this@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);

           
            this.CurrentInput = 0.01 * rand(4,1);
            this.Observation = [this.CurrentInput; 0];
            this.MaxRewardInput = this.CurrentInput;
        end

        %============= STEP =============%
        function [Observation,Reward,IsDone,LoggedSignals] = step(this,Action)

            this.CurrentInput = double(Action);

            
           output_val = this.TargetFunction(this.CurrentInput);

            if ~isfinite(output_val)
                output_val = 1e3;
            end
            
            Reward = 0.01 - output_val;
            
            if ~isfinite(Reward) || ~isreal(Reward)
                Reward = -1e3;
            end


           
            if Reward > this.MaxRewardValue
                this.MaxRewardValue = Reward;
                this.MaxRewardInput = this.CurrentInput;
            end

          
            Observation = [this.CurrentInput; output_val];
            this.Observation = Observation;

           
            IsDone = false;

            LoggedSignals = [];
        end

        %============= RESET =============%
        function InitialObservation = reset(this)
            this.CurrentInput = 0.01 * rand(4,1);
            InitialObservation = [this.CurrentInput; 0];
            this.Observation = InitialObservation;
        end
    end
end
