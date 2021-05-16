Notas a considerar

1) Usa Bcl_knn, NO Bcl_knn_old

1)Para ejecutar el load, se deben poner los archivos en el directorio de trabajo.

2)Se decidió poner en un solo script los 3 datasets. Sin embargo, su ejecución es por separado. Se debe usar
ctr+enter (o cmd+enter en mac) para ir corriendo cada código por separado. 

3) Para el caso del dataset Faces, se usaron dos métodos, y se muestra el accuracy para ambos, nada más que para demostrar algo que 
encontré durante el estudio. Uno utiliza exclusivamente PCA, mientras que el otro SFS con PCA, cuyo tiempo de simulación aumenta
en 50 segundos por la elección de un elevado número de  features mediante SFS. Producto de esos 50 segundos extras, se decidió poner en 
un archivo aparte 'faces.mat' las características ya extraídas, para que así sea más rápida la corrección. 

Indíce:
líneas 5 a 102 FACES con SFS Y PCA (Ejecución ctr+enter o cmd+enter, "parándose en dicha parte del script")
líneas 102 a 188 FACES exclusivamente con PCA (idem)
líneas 190 a 300 TORTILLAS (idem)
líneas 301 a 412 GENDER (idem)