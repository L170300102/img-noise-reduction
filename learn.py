
# from data import *
# from config import *
# from preprocess import *
# from model import *
from main import *

callbacks_list = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath = 'models/baseline_model.h5',
        monitor='val_loss',
        save_best_only=True
    )
]

hist = model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset, callbacks=callbacks_list)

# plt.plot(hist.history["loss"], label='train_loss')
# plt.plot(hist.history["val_loss"], label='val_loss')
# plt.title('loss_plot')
# plt.legend()
# plt.show()

model = tf.keras.models.load_model('models/baseline_model.h5')