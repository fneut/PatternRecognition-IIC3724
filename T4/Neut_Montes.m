clc;
clear all;
close all;
warning off; 
X=load('xdata.mat'); %cargar base de datos

%%%%%%%%%%%%%%% INICIALIZACION DE LA BASE DE DATOS%%%%%%%%%%%%%%%%%%%%%%

Xtrain = X.Xtrain; 
dtrain = X.ytrain;
Xval = X.Xval;
dval = X.yval;
Xtest = X.Xtest;
dtest = X.ytest;
Xn = ['x_1';'x_2'];
%Bio_plotfeatures(Xtrain,dtrain,Xn); %Se analiza como estan siendo repartidas las 2 clases para las 2 features existentes.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%   SVM       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%% INICIALIZACION DE LA BUSQUEDA DE HIPERPARAMETROS %%%%%%%%%%%%%%%%%%%%%%

%%%% A continuacion se muestra todo lo que fue la búsqueda de
%%%% hiperparametros, de tal modo de entrenar a Xtrain y conseguir el
%%%% máximo accuracy posible en validacion, en base a distintos tipos de funciones de Kernel en SVM.
%%%% Se deja comentado para evitar tiempos largos de simulacion al momento de corregir(El profesor me pidió que dejara
%%%% comentado para mostrar evidencia del proceso)



%%%Se seleccionaran distintos kernel del clasificador svm para reportar su
%%%accuracy con el validation data. También se analizaran distintos
%%%clasificadores vistos en tareas anteriores.


%%Descomentar aca si se quiere probar el proceso
% k = 0;
% k=k+1; b(k).name = 'libsvm';   b(k).options.kernel = '-t 0';     % 'SVM - linear'    
% k=k+1; b(k).name = 'libsvm';   b(k).options.kernel = '-t 1' ;    % 'SVM - polynomial'    
% k=k+1; b(k).name = 'libsvm';   b(k).options.kernel = '-t 2';     % 'SVM - rbf' 
% k=k+1; b(k).name = 'libsvm';   b(k).options.kernel = '-t 3';     % 'SVM - sigmoid' 
% %k=k+1; b(k).name = 'knn';   b(k).options.k      = 1;            % 'KNN con k=1'    
% %k=k+1; b(k).name = 'lda';   b(k).options.p      = [];           % 'LDA'    
% %k=k+1; b(k).name = 'maha';  b(k).options        = [];           % 'Mahalanobis'    
% op = b;
% op = Bcl_structure(Xtrain,dtrain,op); %Los distintos clasificadores realizan la clasificacion del training data
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%% TESTING CON VALIDACION%%%%%%%%%%%%%
% 
% ds = Bcl_structure(Xval,op);          %Se clasifica el validation set con los distintos clasificadores elegidos (en base a la clasificacion realizada en training)   
% p  = Bev_performance(ds,dval);        %Se encuentra el accuracy de los distintos clasificadores
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%% RESULTADOS AACURACY VALIDACION CON LOS DISTINTOS KERNEL %%%%%%%%%%%%%%%%%%
% 
% K = {'linear','polynomial','rbf', 'sigmoid'};
% disp('SVM Performance:')
% clc;
% for i=1:k
%     fprintf('%d) %15s %7.4f\n',i,op(i).options.string,p(i))
% end
%%fin de descomentar


%%%%%% FIN ETAPA BUSQUEDA DE HIPERPARAMETROS SVM %%%%%%%%%%%%%




%%%% Ya pasada la etapa de búsqueda de hiperparametros, mediante la cual 
%%%% se obtuvo la funcion kernel a utilizar en el clasificador SVM que maximiza el
%%%% accuracy en validacion, ahora se realizará el proceso de training
%%%% con dicha función kernel, para ver el accuracy obtenido en testing, esperando que sea
%%%% máximo como lo fue en validación.


%%%%%%%%%% TRAINING CON EL KERNEL RBF ENCONTRADO DE HIPERPARAMETROS
%%%%%%%%%% %%%%%%%%%%%%%%%%%%


op_svm_1.kernel = '-t 2'; %op.gamma ='-g 0.1';
op_svm = Bcl_libsvm(Xtrain,dtrain,op_svm_1);



%%%%%%%%%%%TESTING %%%%%%%%%%%%%%%%%%%%%%%%%%%%

ds_svm = Bcl_libsvm(Xtest,op_svm);
ds1_svm = Bcl_libsvm(Xval,op_svm);


%%%%%%%%%%%RESULTADOS %%%%%%%%%%%%%%%%%%%%%%%%%%%%
A1 = Bev_performance(ds_svm,dtest); %accuracy testing 
p_val_svm = Bev_performance(ds1_svm,dval); %accuracy validacion 
fprintf('Accuracy en validacion SVM = %5.4f\n',p_val_svm)
fprintf('A1  = Accuracy en testing SVM = %5.4f\n',A1)




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%   ANN       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%% Inicializacion de búsqueda de hiperparametros ANN %%%%%%%%%%%%%%%%%%%%%%

%%%% A continuacion se muestra todo lo que fue la búsqueda de
%%%% hiperparametros, de tal modo de entrenar a Xtrain y conseguir el
%%%% máximo accuracy posible en validacion, en base a cierto número de
%%%% layers y nodos. Se deja comentado para evitar tiempos largos de
%%%% simulacion al momento de corregir (El profesor me pidió que dejara
%%%% comentado para mostrar evidencia del proceso)

%%Descomentar aca si se quiere probar el proceso
% accurac =[];
% num_nodos_minimo = 5;
% num_nodos_maximo = 20;
% op_redes.method = '3'; %softmax
% for layer_1_nodos = num_nodos_minimo:num_nodos_maximo %nodos layer 1
%     layer_1_nodos
%     for layer_2_nodos = num_nodos_minimo:num_nodos_maximo %nodos layer 2
%         for layer_3_nodos = num_nodos_minimo:num_nodos_maximo
%             op_redes.hidden   = [layer_1_nodos,layer_2_nodos,layer_3_nodos]; %declaro layers y nodos para la iteracion
%             opd = op_redes;
%             opd = Bcl_ann(Xtrain,dtrain,opd); %Encontrar los pesos tal que el error entre la salida y la clasificacion ideal sea minimo
%             ds1 = Bcl_ann(Xtrain,opd); 
%             ds2 = Bcl_ann(Xval,opd); %Salida de la red neuronal utilizando los pesos encontrados en training, y como input Xval
%             p1 = Bev_performance(ds1,dtrain); 
%             p2 = Bev_performance(ds2,dval); %Accuracy sobre validacion. Idealmente hay que encontrar el mínimo de error entre la salida de la red neuronal y validacion (en base a los pesos calculados en training)
%             accurac = [accurac;[p2,layer_1_nodos,layer_2_nodos, layer_3_nodos]]; %Guardo en una matriz la performance obtenida, como a la vez el número de nodos y layers para el cual se produjo dicha performance
%         end
%     end
% end   
% 
% %%%% A continuacion se procede a encontrar la máxima perfomance dentro de
% %%%% toda la iteracion realizada previamente, para asi obtener los
% %%%% hiperparámetros que maximicen el accuracy obtenido en validación, y
% %%%% esperar que también maximicen el accuracy a obtener en testing.
% maximizar_performance = 0;
% num_fila = 0;
% for k=1:length(accurac)
%     k;
%     valor = accurac(k,1);
%     if(valor > maximizar_performance)
%         maximizar_performance=valor; %Se va buscando el maximo valor de performance en validacion
%         nodos_layer_1 = accurac(k,2); %Se guarda el numero de nodos en layer_1
%         nodos_layer_2 = accurac(k,3); %Se guarda el numero de nodos en layer_2
%         nodos_layer_3 = accurac(k,4); %Se guarda el numero de nodos en layer_3
%     end  
% end
%%Fin descomentar


%%%%%% FIN ETAPA BUSQUEDA DE HIPERPARAMETROS ANN %%%%%%%%%%%%%

%%%% Ya pasada la etapa de búsqueda de hiperparametros, mediante la cual 
%%%% se obtuvo el numero de layers y el numero de nodos que maximizan el
%%%% performance en validacion, ahora se realizará el proceso de training
%%%% con dicho número de layers y nodos encontrados previamente, para
%%%% después ver el accuracy obtenido en testing, esperando que sea
%%%% máximo como lo fue en validación.



%%%% Es importante recalcar que como el algoritmo se inicializa con pesos
%%%% de forma aleatoria, los resultados del algoritmo van cambiando.
%%%% Es por ello que se procederá a encontrar el promedio de 10
%%%% iteraciones, y reportar el accuracy en testing y validacion.


accurac =[];
op_ann.method = '3'; %softmax
op_ann.hidden   = [10 13 9]; %nodos y layers que maximizan el accuracy de validacion. Valores encontrados en la búsqueda de hiperparámetros.
for prom = 1:10
opd = Bcl_ann(Xtrain,dtrain,op_ann); %Encontrar los pesos tal que el error entre la salida de la red neuronal y la clasificacion ideal sea minimo

ds1 = Bcl_ann(Xval,opd);%Salida de la red neuronal utilizando los pesos encontrados en training, y como input Xval
ds2 = Bcl_ann(Xtest,opd);%Salida de la red neuronal utilizando los pesos encontrados en training, y como input Xtest

p1 = Bev_performance(ds1,dval); %Accuracy sobre validacion. Idealmente hay que encontrar el mínimo de error entre la salida de la red neuronal y validacion (en base a los pesos calculados en training)
p2 = Bev_performance(ds2,dtest); %Accuracy sobre testing. Idealmente hay que encontrar el mínimo de error entre la salida de la red neuronal y testing (en base a los pesos calculados en training)
accurac = [accurac;[p1, p2]]; %Guardo en una matriz la performance obtenida en validacion y testing respectivamente, de la iteracion respectiva.
end 

%%% Ya realizada la matriz, se procede a encontrar el promedio.
p_val_redes = mean(accurac(:,1));%accuracy validacion obtenido del promedio de 10 iteracione
A2 = mean(accurac(:,2)); %accuracy testing obtenido del promedio de 10 iteraciones
fprintf('Accuracy en validacion Redes neuronales = %5.4f\n',p_val_redes)
fprintf('A2  = Accuracy en testing Redes neuronales = %5.4f\n',A2)





%%%%%%%%%%%%%%%%%%%%%%%%%%DECISION LINE ANN
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% bcl(1).name = 'ann';  %ANN
% bcl(1).options.hidden   = [10 13 9];
% opd = Bcl_structure(Xtrain,dtrain,bcl);
% figure
% Bio_decisionline(Xtest,dtest,['x1';'x2'],opd);
% title('testing data and decision lines')


A1
A2