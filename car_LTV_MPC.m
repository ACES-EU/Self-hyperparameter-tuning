function results = car_LTV_MPC(TM,X_scene)
% X_scene: parameters used to generate scenario
% TM: MPC controller parameters to be tunned
% 'mpc_solve.p' is used to solve the mpc problem 


isLTV=1; % 1 = LTV-MPC, 0 = LPV-MPC
isYmin=1; % 1 = add output constraint
isRefpreview=1; % 1= with reference preview; 0 = w/o ref. prev.
verbose = 0;

y_scene = [0 3]';
T_sim = 30; % simulation time

num_vehicle = size(X_scene,1)/2;
if num_vehicle ~= ceil(num_vehicle) || num_vehicle ~= size(y_scene,1)
    error('Please double check the definition of X_scene and y_scene')
end

lane_label_obs = zeros(num_vehicle,1);
for k=1:num_vehicle
    if y_scene(k) ==0
        lane_label_obs(k) = 0;
    else
        lane_label_obs(k) =1;
    end
end

L=4.5;  % distance between front and rear axle
Lw=L/6; % wheel width, assume to be 1/6 of L
Ts=TM(1); % sample time [s]
vmin=1/3.6; % min velocity [m/s]
vmax_org=90/3.6; % max velocity [m/s]
vmax = vmax_org;
deltamin=-pi/4; % min steering angle [rad]
deltamax=pi/4; % max steering angle [rad]

road_width = 6; % the width of the road [m] 
w_car = 1.8; % the width of the car [m];

nx=3; % number of states
nu=2; % number of inputs
% ny=3; % number of outputs, full state estimation is assumed

p=ceil(TM(3)); % prediction horizon
m=ceil(TM(2)*TM(3));  % control horizon

x0=[0;0;0]; % initial state
if x0(2) ==0
    lane_label_ego =0;
else
    lane_label_ego =1;
end

vref= 50/3.6; %[m/s]
yref=0; % [m]
useq=[ones(p,1)*vref,zeros(p,1)]; % initial nominal input trajectory

Tstop=T_sim; % Simulation time [s]
N=ceil(Tstop/Ts/2)*2;

% output reference
s=vref*Ts*(0:N+p-1)';
ref_nominal = [s(:),yref*ones(N+p,1),zeros(N+p,1)];
ref= ref_nominal;

% Specify the condition of the moving obstacle (assumed to be same for all obstacle vehicles)
Lobs = L;
Lobsw=Lobs/6; % wheel width, assume to be 1/6 of Lobs
theobs = 0; % move horizontally for all obstacle vehicles

% Extract values (initial longitudianl position and velocity) from X_scene
x_obs_f = zeros(num_vehicle,1);
x_obs_r = zeros(num_vehicle,1);
v_obs  = zeros(num_vehicle,1);
y_obs_f = y_scene; % [m]

for k = 1:num_vehicle
   x_obs_f(k) = X_scene(k*2-1); % [m]
   x_obs_r(k) = x_obs_f(k)- Lobs*cos(theobs);
   v_obs(k) = X_scene(k*2)/3.6; %[m/s]
end


% MPC weights
% weights (penalty = ||wt*var(k)||_2^2)
uwt=[1 1]; 
duwt=[10^TM(4) 10^TM(5)];
% duwt=[TM(4) TM(5)];
xwt=[0 10 1];
rhoeps=1e3; % weight on slack variable used for soft-constraints

% constraints
umin=[vmin;deltamin];
umax=[vmax;deltamax];
dumin=-Inf(2,1); dumax=Inf(2,1);
accel_lim_vehicle = 4; % [m/s^2]
dumin(1) = -accel_lim_vehicle*Ts;
dumax(1) = accel_lim_vehicle*Ts;
dumin(2) = -60/180*pi*Ts;
dumax(2) = 60/180*pi*Ts;

% smin = -Inf(p,nx);
% smax = Inf(p,nx);
smin_org = -Inf(p,nx);  
smax_org = Inf(p,nx);
ymin_org =-road_width/4 + w_car/2 ;  % Note: here y is the lateral position
ymax_org = road_width - w_car/2-road_width/4;
% smin_org(:,1) = 0;
smin_org(:,2) = ymin_org;
smax_org(:,2) = ymax_org;
smin_org(:,3) = -pi/2;
smax_org(:,3) = pi/2;

t_ttc = Ts*(m+1);

% variables used for plotting
X_mpc=zeros(N,nx);
U=zeros(N,nu);
X_obs = zeros(N,2*num_vehicle);

% Initial 
x_mpcstate=x0;
uold=useq(1,:)';
uref=[vref;0];
U(1,:) = uold;

% Specify parameters for lane change
% xsafe = 2*uold(1) ; % safety distance between two cars (longitude)
xsafe = 10;
ysafe = 3;

% collision detection & lateral safety index
ind_collision = zeros(N,num_vehicle); ind_collision_combined = [];
lane_change = zeros(num_vehicle,1);
ind_infes_solver = [];

% check calculation time for each time step
tictoc_overall = zeros(N,1);

for t=1:N
    tic
    % Save results for plotting
    xr=x_mpcstate(1)-L*cos(x_mpcstate(3));
    xf=x_mpcstate(1);
%     yr=x_mpcstate(2)-L*sin(x_mpcstate(3));
    yf=x_mpcstate(2);
 
    smin = smin_org; smax = smax_org;

    if isYmin % output, set additional
        % calculate the absolute longitudinal distance between each obstacle vehicle and ego vehicle
        x_absolute_safe =abs(x_obs_f - xf)-(xsafe + L);
        is_dangerous_lang = (x_absolute_safe<=0);
        
        for k = 1:num_vehicle
            constraints_obstacle = update_constraints_mpc(xf,xr,yf,x_obs_f(k),x_obs_r(k),y_obs_f(k),uold(1),v_obs(k),ysafe,L,road_width,w_car,t_ttc,lane_change(k),k,is_dangerous_lang,p,Ts,lane_label_ego,lane_label_obs(k));
            lane_change(k) = constraints_obstacle.lane_change;
            lane_label_ego = constraints_obstacle.lane_label_ego;
            xmin_update = constraints_obstacle.xmin;  % Note: here, x is the longitudinal position of the obstacle vehicle
            xmax_update = constraints_obstacle.xmax;
            ymin_update = constraints_obstacle.ymin;
            ymax_update = constraints_obstacle.ymax;
            if smin(:,1) < xmin_update
                smin(:,1) = xmin_update;
            end
            if smax(:,1) > xmax_update
                smax(:,1) = xmax_update;
            end
            if smin(1,2) < ymin_update && is_dangerous_lang(k)
                smin(:,2) = ymin_update;
            end
            if smax(1,2) > ymax_update
                smax(:,2) = ymax_update;
            end
        end
        if (sum(is_dangerous_lang) <1) && (lane_label_ego)
            smin(:,2) =3;
        end

    end 

    if smin(1,2) >0
        ref_update_y = smin(:,2);
    else
        ref_update_y = zeros(p,1);
    end
    ref(t+1:t+p,2) = ref_update_y;



    if isRefpreview
        % with reference preview
        r=ref(t+1:t+p,:); % xref = yref (full state observation)
    else
        % without reference preview
        r=ones(p,1)*ref(t+1,:);
    end
    xref =r;
    
    [useq,Flag(t)]=mpc_solve(p,m,x_mpcstate,uold,useq,isLTV,duwt,xwt,uwt,rhoeps,...
            umin,umax,smin,smax,dumin,dumax,xref,uref,L,Ts);

    if Flag(t) == -2
        ind_infes_solver = [ind_infes_solver;t];
        msg_inf = 'MPC solver not able to find a feasible solution, a collision can happen';
        if verbose
            disp(msg_inf)
        end
    end
    
    u = useq(1,:)';
    
    % Store for ploting
    X_mpc(t,:)=x_mpcstate';
    U(t,:)=u';

    for k = 1:num_vehicle
        X_obs(t,2*k-1) = x_obs_f(k);
        X_obs(t,2*k) = y_obs_f(k);
    end


    % Check if collision actually happened with the computed u
    for k = 1:num_vehicle
        ind_collision(t,k) = collision_check(xf,x_obs_f(k),yf,y_obs_f(k),L,w_car,k,t,verbose);
    end

    if sum(ind_collision(t,:)) >0
        ind_collision_combined = [ind_collision_combined; t];
    end

    % state update of car
    x_mpcstate=car_model(x_mpcstate,u+[0*randn(2,1)],0,L,Ts);
    
    % state update of obstacle
    for k = 1:num_vehicle
        x_obs_f(k) = x_obs_f(k) + v_obs(k) *Ts;
        x_obs_r(k) = x_obs_f(k) - Lobs*cos(theobs);
    end

    uold = u;
    
    % update nominal sequence
    useq=[useq(2:p,:);useq(p,:)]; % shift optimal sequence

    tictoc_overall(t) = toc;
end
% tictoc = mean(tictoc_overall);
tictoc = max(tictoc_overall); 
V_nominal = vref*ones(N,1);
results = struct('X_mpc',X_mpc,'U',U,'X_obs',X_obs,'num_vehicle',num_vehicle,...
                 'tictoc',tictoc,'ref',ref(1:N,:),'TM',TM,'L',L,'Lobs',Lobs,...
                 'T_sim',T_sim,'w_car',w_car,'t_ttc',t_ttc,'ind_collision_combined',ind_collision_combined,...
                 'y_safe',ysafe,'ind_infes_solver',ind_infes_solver,...
                 'ind_collision',ind_collision,'X_scene',X_scene,'V_nominal',V_nominal);


end
%%
function constraints_obstacle = update_constraints_mpc(x_ego_f,x_ego_r,y_ego_f,x_obs_f,x_obs_r,y_obs_f,v_ego,v_obs,ysafe,L,road_width,w_car,t_ttc,lane_change,k,is_dangerous_lang,p,Ts,lane_label_ego,lane_label_obs)
% specify the constraints for  each obstacle
    if ((x_obs_f > x_ego_f && x_obs_r - x_ego_f <= max(t_ttc *(v_ego-v_obs),L)) || (x_obs_f < x_ego_f && x_ego_r - x_obs_f <= max(t_ttc *(v_obs-v_ego),L))) && (abs(y_ego_f-y_obs_f) <w_car)
            isCollision = 1;
    else
        isCollision = 0;
    end 

    is_dangerous_lang_others = is_dangerous_lang;
    is_dangerous_lang_others(k) =[];

    if abs(y_ego_f-road_width/2) <=0.02 || y_ego_f-road_width/2 >=0
        lane_label_ego = 1;
    elseif abs(y_ego_f)<0.02
        lane_label_ego = 0;
    end

    %     lane_diff = lane_label_ego-lane_label_obs;
    lane_diff = (abs(y_ego_f-y_obs_f) >w_car);

    if is_dangerous_lang(k) && (~lane_diff) % additional constraints are needed when the ego and obstacle vehicles are on within safety distance both laterally and longitudinally.
        % ego and obstacle vehicles are on the same lane
        % if ego and obstacle vehicles are on different lanes, no additional constraitns are needed 
        if (sum(is_dangerous_lang_others) <1) && (~isCollision || lane_change) && (x_obs_f > x_ego_f) % if (all the other obstacle vehicles are outside of safety distances) and (No collisison expected) and (obstacle is ahead of ego vehicle)
             % change lane
            if ~lane_label_ego && ~lane_label_obs
                ymin = ysafe;
                ymax = Inf;
            else
                ymin = -road_width/4 + w_car/2;
                ymax = 0;
            end
            lane_change = 1;
            xmin =-Inf(p,1); xmax =Inf*ones(p,1);
        else % brake or accelerate on the same lane (depend on the relative position of the ego and obstacle vehicles)
            if ~lane_label_ego
                if y_ego_f > w_car/2
                    ymin = ysafe;
                    lane_change = 1;
                else
                    ymin = -road_width/4 + w_car/2;
                end
            else
                if y_ego_f < w_car/2
                    ymin = -road_width/4 + w_car/2;
                else
                    ymin = ysafe;
                end
            end
            ymax = Inf;
            Xobs_pred_long_f = linspace(x_obs_f+ v_obs*Ts, x_obs_f + v_obs*Ts*(p+1),p)';
            if x_obs_f > x_ego_f
                xmax(:,1) = Xobs_pred_long_f-1.1*L;
                xmin =-Inf(p,1);
            else
                xmin(:,1) = Xobs_pred_long_f + 1.1*L;
                xmax =Inf*ones(p,1);
            end 
        end
    elseif lane_label_ego
        if lane_label_obs && lane_diff
            ymin =-road_width/4 + w_car/2; ymax = Inf; xmin =-Inf(p,1); xmax =Inf*ones(p,1);
        else 
            ymin =3; ymax = Inf; xmin =-Inf(p,1); xmax =Inf*ones(p,1);
        end
    elseif ~lane_label_ego && ~lane_label_obs && lane_diff
        ymin =3; ymax = Inf; xmin =-Inf(p,1); xmax =Inf*ones(p,1);
    else
        ymin =-road_width/4 + w_car/2; ymax = Inf; xmin =-Inf(p,1); xmax =Inf*ones(p,1);
    end

    constraints_obstacle = struct ('lane_change',lane_change,'lane_label_ego',lane_label_ego,...
                                    'ymin',ymin,'ymax',ymax,'xmin',xmin,'xmax',xmax);

end


function ind_collision_t = collision_check(x_ego_f,x_obs_f,y_ego_f,y_obs_f,L,w_car,k,t,verbose)
% Check if collision actually happened with the computed u
    if (abs(x_ego_f-x_obs_f) <= L) && (abs(y_ego_f-y_obs_f) <w_car)
        ind_collision_t = 1;
        if verbose
            fprintf('A collision happened with Obstacle vehicle %d at time step %d\n',k,t);
        end
    else
        ind_collision_t = 0;
    end

end

    