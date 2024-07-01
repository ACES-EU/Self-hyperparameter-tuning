function [xnext,y,A,B,C]=car_model(x,u,returnModel,L,Ts)
% Bicycle model, rear-axle drive, front-axle reference 
%
% (C) 2019 A. Bemporad, September 18, 2019

nx=3;
nu=2;

th=x(3);
v=u(1);
delta=u(2);
cth=cos(th+delta);
sth=sin(th+delta);
sdel=sin(delta);
cdel=cos(delta);

xdot=[v*cth;v*sth;v/L*sdel];
y=x;

xnext=x+Ts*xdot;

A=zeros(nx,nx);
B=zeros(nx,nu);
C=eye(nx);

if returnModel
    Ac=[0 0 -v*sth;
        0 0 v*cth;
        0 0 0];
    
    A=eye(nx)+Ts*Ac;
    
    Bc=[cth -v*sth;
        sth v*cth;
        sdel/L v/L*cdel];
    
    B=Ts*Bc;
end