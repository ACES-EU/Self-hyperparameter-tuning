function f_i = benchmark_mpc_calibration(TM,i)
% Benchmark (MPC calibration task) for D-GLIS
% Inputs: 
%       TM: optimization variables (mpc parameters to calibrate in this case)
%           dimension: N by nvars  (N is the number of TM for testing)
%       i: the identity of the agent
% Output:
%       f_i: the objective fun. eval. of agent i
%            (in this case, 4 agents are available)

% Note, the output is scaled bewtween 0 and 1
    agent = i;
    verbose = 1;
    nvars = 5;
    [n,m]=size(TM);
    if nvars==1
        n=max(m,n);
    end
    f_i=ones(n,1);
    for i = 1:n
        switch agent
            case 1  % for agent 1
                X_scene = [10   38   33   40 ]';
                [f_mult1,f_mult2,f_mult3,f_mult4] = f_mult(TM(i,:),i,X_scene,verbose);
                f_i(i) = 1-exp(-(0.5*f_mult1 + 0.5*f_mult2 + f_mult4));
                %f_i(i) = 5*(1-exp(-(0.5*f_mult1 + 0.5*f_mult2  + f_mult4)))-2.5;
    
            case 2  % for agent 2
                X_scene = [15   30   17   48 ]';
                [f_mult1,f_mult2,f_mult3,f_mult4] = f_mult(TM(i,:),i,X_scene,verbose);
                f_i(i) = 1-exp(-(0.8*f_mult1 + 0.2*f_mult2 + f_mult4));
                %f_i(i) = 5*(1-exp(-(0.8*f_mult1 + 0.2*f_mult2 + f_mult4)))-2.5;
    
            case 3  % for agent 3
                X_scene = [20   40   60   42]';
                [f_mult1,f_mult2,f_mult3,f_mult4] = f_mult(TM(i,:),i,X_scene,verbose);
                f_i(i) = 1-exp(-(0.3*f_mult1 + 0.7*f_mult2 + f_mult4));
                %f_i(i) = 5*(1-exp(-(0.3*f_mult1 + 0.7*f_mult2   + f_mult4)))-2.5;

            case 4  % for agent 4
                X_scene = [9   60   20   45 ]';
                [f_mult1,f_mult2,f_mult3,f_mult4] = f_mult(TM(i,:),i,X_scene,verbose);
                f_i(i) = 1-exp(-(0.6*f_mult1 + 0.4*f_mult2 + f_mult4));
                %f_i(i) = 5*(1-exp(-(0.6*f_mult1 + 0.4*f_mult2  + f_mult4)))-2.5;
        end
    end
end


function [f_mult1,f_mult2,f_mult3,f_mult4] = f_mult(TM,i,X_scene,verbose)
    results = car_LTV_MPC(TM,X_scene);
    U_mpc = results.U;
    V_nominal = results.V_nominal;
    tictoc = results.tictoc;
    ind_collision_combined = results.ind_collision_combined;
    ind_infes_solver = results.ind_infes_solver;
    if ~isempty(ind_collision_combined)
        Iscollision=1;
    else
        Iscollision=0;
    end
    f_mult1 = mean(abs(U_mpc(:,1)-V_nominal(:))./V_nominal(:)); % variation in velocity
    f_mult2 = mean(abs(U_mpc(:,2))./0.1); % variation in steering angle
    f_mult3 = 1000*max(0,tictoc-TM(1));  % feasibility of the solver 
    f_mult4 = 1000*Iscollision;  % penality for collision
    if verbose
        if ~isempty(ind_collision_combined) || ~isempty(ind_infes_solver)
            fprintf('collision occurs \n')
        end
    end

end