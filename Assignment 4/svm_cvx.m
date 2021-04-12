clc
clear
close all

addpath('/Users/abhijith/Desktop/TECHNOLOGIE/Resources/Codes/cvx')

%% IMPORT DATA

T = readtable('a4.csv');
table = T{:,:};

%% SET

X = table(1:2,:);
labels = table(3,:);

cvx_begin
    variables m(2) t(10)
    minimize sum(t)
    for k = 1:10
        X(:,k) - m <= t(k)
        m - X(:,k) <= t(k)
    end
cvx_end

%%

cvx_begin
    variables m(2) t(10)
    minimize sum(t)
    for k = 1:10
        X(:,k) - m >= -t(k)
        X(:,k) - m <= t(k)
    end
cvx_end

%%

% m = [0.137;.924];
% sum(vecnorm(X-m,Inf,1))

% A = [0.8147 0.6324 0.9575 0.9572; 0.9058 0.0975 0.9649 0.4854; 0.1270 0.2785 0.1576 0.8003; 0.9134 0.5469 0.9706 0.1419];
% b = [0.4218; 0.9157; 0.7922; 0.9595]

X0 = [X(:,2)'; X(:,6)'];
