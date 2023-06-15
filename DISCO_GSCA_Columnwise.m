function Results=DISCO_GSCA_Columnwise(X,Target,T,lambda,convergence, maxit)
% Results=DISCO_GSCA_Columnwise(X,Target,T,lambda,convergence, maxit)
%
% Input:
% - X = concatenated variable-wise linked data blocks
% - Target = target matrix for the concatenated data
% - T = starting values for the score matrix T
% - lambda = penalty parameter lambda
% - convergence = convergence criterion
% - maxit = maximum number of iterations
%
% Output: Results -> structure containing
% - T = obtained score matrix
% - P = obtained loading matrix
% - Penalty = penalty term (lambda*||W.*(T - Target)||^2)
% - Missfit = misfit (||X - T*P'||^2)
%
% (c) 2011-11-26

%% Analysis
%weight matrix
W=ones(size(Target))-Target;
%number of components
NCOMP=size(Target,2);
%starting values
P=X'*T;
%current loss-value q_c
q_c=calculate_q(X,T,P,W,lambda);

iter=0;
stop=0;
while stop==0       
    %optimisation T, given P
    w=max(max(W));
    TargetSter=T- (1/(w^2))*W.*W.*T;
    A= P'*X'+lambda*(w^2)*TargetSter'; 
    [u s v]=svd(A);       
    T=v(:,1:NCOMP)*u(:,1:NCOMP)';           
    %optimisation P, given T 
    P=X'*T;
    %updated loss-value q_u
    q_u=calculate_q(X,T,P,W,lambda);        
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
%rotate components with the same status 
[a b]=size(Same_Status);
for i=1:a
    %rotate scores
    Trot_same_status=T(:,Same_Status{i,1}); 
    [Trot_same_status_rotated, Rotation_Matrix] = ... 
        rotatefactors(Trot_same_status,'Normalize','off'); 
    %rotate components
    Prot_same_status=P(:,Same_Status{i,1});    
    Prot_same_status_rotated= Prot_same_status*Rotation_Matrix;
    %store results
    T(:,Same_Status{i,1})=Trot_same_status_rotated;
    P(:,Same_Status{i,1})=Prot_same_status_rotated;
end

%% Save output
Results.T=T;
Results.P=P;
Results.lambda=lambda;
Results.Penalty=lambda*ssq(W.*T);
Results.missfit=ssq(X-T*P');

%% Help functions
    function Loss=calculate_q(X,T,P,W,lambda)
        %Loss=calculate_q(X,T,P,W,lambda)
        %returns loss value
        D=X-T*P';
        B=W.*T;
        Loss=sum(sum(D.*D))+lambda*sum(sum(B.*B));
    end
    function SS = ssq (X)
        %SS = ssq(X);
        %returns the sum of squares of X
        SS=X(:)'*X(:);
    end

end

