function Y= reconocedorSC(X)%Se asume como input LA MATRIZ BINARIA de la imagen
[filas,columnas] =size(X);%Extrae el número de filas y columnas de la matriz X. 
y=X; %Copia la matriz X en una nueva variable "y".
for j=1:columnas % Nos "paramos" en una columna específica.
    filas_unos_encontrados = find(y((1:filas),j)==1);% Entrega un "array" con los números de las filas que tengan unos, para la columna "j" de la matriz X.
    for n=1:length(filas_unos_encontrados)-1 
        %Se irá iterando sobre pares de unos hallados en la columna "j" de la matriz X, con el objetivo de encontrar ceros entremedio de ese par y rellenarlos con "unos". 
        buscar_ceros_entre_unos = find(y(filas_unos_encontrados(n)+1 : filas_unos_encontrados(n+1)-1 , j) == 0);%Busco si existen o no ceros dentro del par de unos en el que se esté iterando.
        chequear_si_hay_ceros = not(isempty(buscar_ceros_entre_unos)); %Si es 1, significa que hay ceros entremedio de algun par de unos.
        if(chequear_si_hay_ceros==1)             
            y(filas_unos_encontrados(n)+1:filas_unos_encontrados(n+1)-1,j)=1;%Cambio los ceros encontrados a unos.
        end
%Si no encontré ceros dentro del par de unos anterior, buscaré si existen ceros en el
%siguiente par de unos que exista en la columna (en caso de existir).
    end
end
Y2 = xor(y,X); %Este XOR entragará la matriz Y2 se entrega una nueva matriz cuyos 1 corresponden EXCLUSIVAMENTE a las columnas rellenados previamente con unos, desapareciendo a su vez los pixeles de la letra
[~,islas] = bwlabel(Y2,4); %Este comando entrega la cantidad de objetos o huecos detectados en la matriz Y2. 
if(islas>=2) %Si se detectan 2 objetos o huecos en la matriz Y2, entonces es una S.(se pone el >= para prevenir más opciones que Y=1 y Y=0)
   Y=1; 
elseif(islas<=1) %Si se detecta 1 objeto en la matriz Y2, entonces es una C. (misma analogía con el <=)
   Y=0;
end
end