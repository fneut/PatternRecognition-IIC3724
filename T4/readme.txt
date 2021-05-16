Simplemente ejecutar run y se mostrará el accuracy A1 y A2. Gran parte del código
se encuentra comentado, lo cual corresponde a la búsqueda de los hiperparámetros mediante
el conjunto validación. Le pregunté al profesor si lo debía incluir o no, y me dijo vía
mail que lo entregara comentado, para que de esta forma quede evidenciado todo el proceso
de búsqueda que se realizo. Aún así, si se quiere probar la búsqueda efectuada simplemente
se debe descomentar desde la línea 41 a la línea 65 para SVM, mientras que de la 
línea 121 a la línea 156 para redes neuronales (igualmente sale específicado en el código que se debe descomentar).

Se comienza el código con SVM desde el título en la línea 21, y contínua hasta la línea 100. 
Luego, comienza ANN con el título en la línea 107, y contínua hasta la línea 195.

Muchas gracias!



Notas adicionales.
-Se debe añadir Neut_montes.m al directorio donde se encuentre la base de datos 'xdata.mat'.
-Se utiliza Matlab R2016b
-Se utiliza Bcl_ann() y Bcl_libsvm. 
