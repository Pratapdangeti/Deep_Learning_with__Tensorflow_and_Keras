import caption_model_generator
from keras.callbacks import ModelCheckpoint

# CSV loggers are for writing down runtime statistics into logging file
from keras.callbacks import CSVLogger
csv_logger = CSVLogger('log_file.CSV', append=True, separator=';')


# Function for model training using generators
def model_training(weight = None, batch_size=32, epochs = 10):
    # Calling the function
    cmg = caption_model_generator.CaptionModelGenerator()
    model = cmg.create_final_model()

    if weight!= None:
        model.load_weights(weight)
    # Name of models to be created at each epoch
    file_name = 'weights-epoch-{epoch:02d}.hdf5'
    checkpoint = ModelCheckpoint(file_name, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint,csv_logger]
    # Fitting the model using generator
    model.fit_generator(cmg.data_generator(batch_size=batch_size), steps_per_epoch=cmg.total_samples/batch_size, epochs=epochs, verbose=2, callbacks=callbacks_list)
    try:
        model.save('Models/WholeModel.h5', overwrite=True)
        model.save_weights('Models/Weights.h5',overwrite=True)
    except:
        print ("Error: Unable to save model.")
    print ("Training completed\n")


if __name__ == '__main__':
    model_training(batch_size=128, epochs=50)



