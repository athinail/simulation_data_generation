Repository to create data for object detection. It creates a training set with circle and squares and it creates a test set with circle with sort of corners and a square with more curvy corners.
Repository structure:
simulation_data_generation
|--Data -> the folder where the generated data annotations are saved. 
    |-- Annotations -> the same as above
            |-- test -> contains the folders with the annotations for the test set. The structure 				 in each of the folders contained in the test are according to Mathieu's 				 thesis structure, so that the data are applicable to his code
            | --train ->contains the folders with the annotations for the train set. The structure 				 in each of the folders contained in the test are according to Mathieu's 				 thesis structure, so that the data are applicable to his code
            |--YOLOCuratedData -> contains the structure needed for the annotations in YOLOv6 					   repository
|--data_processing ->
    |--copy_images_and_annotations.py -> copies the images and annotations to the folders 						according to YOLOv6 structure (images, labels , train, val 						etc.)
|--simulation -> contains the scripts for generating the data
    |-- folder_rename.py -> renames the folders in the test and train folder to 0,1,2 etc. to 				     comply with the naming of the respective folders in mathieu's 				     implementation
    |--YOLO_anotations_extract2.py -> contains the class to extract YOLO labels during generating 				               the data annotations
    |-- utils.py _> contains methods to process the data
    |-- test.py -> an initial script that I used to experiment with the libraries to create shapes
    |--model_accuracy.py -> script that defines the std of noise that can be added to the 				     simulation siglans to simulate the "measured signals" 
    |-- ShapeDynamicsSimulator.py --> contains the class that creates the shapes (circle, 						square..etc.) which follow a specific motion (gravity, 						damper-spring etc.)
