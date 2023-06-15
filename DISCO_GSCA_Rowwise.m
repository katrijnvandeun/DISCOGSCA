function Results = DISCO_GSCA_Rowwise(X,Target,T,lambda,convergence, maxit)
% Results=DISCO_GSCA_Rowwise(X,Target,T,lambda,convergence, maxit)
%
% Input:
% - X = concatenated row-wise linked data blocks
% - Target = target matrix for the concatenated data
% - T = starting values for the score matrix T
% - lambda = penalty parameter lambda
% - convergence = convergence criterion
% - maxit = maximum number of iterations
%
% Output: Results -> structure containing
% - T = obtained score matrix
% - P = obtained loading matrix
% - Penalty = penalty term (lambda*||W.*(P - Target)||^2)
% - Missfit = misfit (||X - T*P'||^2)
% 
% The DISCO-GSCA algorithm for the case of rowwise linked data 
% blocks makes use of (an to our setting adapted version of) the 
% Grey Component Analysis algorithm (Thanks to Johan A. 
% Westerhuis for providing the MATLAB code). 
%
% (c) 2011-11-26

%% Analysis
%loadings
P=X'*T;
%number of components
NCOMP=size(Target,2);
%weight matrix
W=ones(size(Target))-Target;
%compute q_c
q_c=calculate_q(X',P,T,W,lambda);

%% adapted version of the (fast) GCA algorithm
%transpose data matrix
X=X';

[I,J] = size(X);
[I,R] = size(Target);

As = sparse(W);
A0 = sparse(Target);
Ab = sparse(diag(As(:)));
Ab = sparse(Ab'*Ab);
Xv = X(:);
A0v = Target(:);A0v = sparse(A0v);

iter=0;
stop=0;
while stop == 0
    %optimisation T, given P (Kiers, 2002)     
    B= P'*X;  
    [u s v]=svd(B);       
    T=v(:,1:NCOMP)*u(:,1:NCOMP)';     
    %optimisation P, given T (Westerhuis, 2007)
    F = (kron(T,speye(I))); F = sparse(F);
    FF = F'*F; FF = sparse(FF);
    FFDD = (FF+lambda*Ab);FFDD = sparse(FFDD); clear FF
    Av = FFDD\(F'*Xv);
    P = reshape(Av',I,R); 
    clear Av FFDD F
    %updated loss-value q_u
    q_u=calculate_q(X,P,T,W,lambda);    
    %difference in loss-values between q_c and q_u 
    Diff=q_c-q_u;
    %judgement    
    if iter >= maxit
       stop=1;
       fprintf('Maximum number of iterations reached\n')
    elseif Diff < 0
       stop=1;
       iter = iter + 1;  
       fprintf('Iteration Nr: %3.0f | q_u: %5.4f\n',iter,q_u);
       fprintf('Divergence\n')
    elseif Diff<convergence
       stop=1;
       iter = iter + 1;  
       fprintf('Iteration Nr: %3.0f | q_u: %5.4f\n',iter,q_u);
       fprintf('Convergence\n')
    else
       q_c = q_u;
       iter = iter + 1;  
       fprintf('Iteration Nr: %3.0f | q_u: %5.4f\n',iter,q_u);
    end 
end

%% Extra rotation of the components with the same status
%find component with the same status
for i=1:NCOMP
    s=Target-repmat(Target(:,i),1,NCOMP);
    Same_Status{i,1}=find(sum(abs(s))==0);
end   
%rotate components with the same status. 
[a b]=size(Same_Status);
for i=1:a
    %rotate loadings
    Prot_same_status=P(:,Same_Status{i,1}); 
    [Prot_same_status_rotated, Rotation_Matrix] = ... 
        rotatefactors(Prot_same_status,'Normalize','off');
    %rotate components
    Trot_same_status=T(:,Same_Status{i,1});    
    Trot_same_status_rotated= Trot_same_status*Rotation_Matrix;
    %store results
    T(:,Same_Status{i,1})=Trot_same_status_rotated;
    P(:,Same_Status{i,1})=Prot_same_status_rotated;
end

%% Save output
Results.T=T;
Results.P=P;
Results.lambda=lambda;
Results.Penalty=lambda*ssq(W.*P);
Results.missfit=ssq(X'-T*P');

%% Help functions
    function Loss=calculate_q(X,T,P,W,lambda)
        %Loss=calculate_q(X,T,P,W,lambda)
        %returns loss value
        D=X-T*P';
        B=W.*T;
        Loss=sum(sum(D.*D))+lambda*sum(sum(B.*B));
    end
    function SS = ssq (X)
        % SS = ssq(X);
        % returns the sum of squares of X
        SS=X(:)'*X(:);
    end

end