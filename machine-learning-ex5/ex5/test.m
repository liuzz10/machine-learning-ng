function [ error_test ] = test(X_poly, y, X_poly_test, ytest, lambda )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

lambda = 3;

[theta] = trainLinearReg(X_poly, y, lambda);

error_test = linearRegCostFunction(X_poly_test, ytest, theta, 0);

end

