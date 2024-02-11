import source.Time_Series_Reading_Codes.read_data as rd
import source.Time_Series_Reading_Codes.normalize_per_user as nu
import source.Time_Series_Reading_Codes.global_normalize as ng
import source.Time_Series_Reading_Codes.generate_images as gi

window_size = 300
data = rd.read_data('ankle')

train_users = ['GOTOV06','GOTOV16','GOTOV17','GOTOV18','GOTOV20','GOTOV24']
val_users= ['GOTOV05']
test_users = ['GOTOV08']

ankle_train,ankle_val,ankle_test = nu.generate_user_normalised_data(data,train_users,val_users,test_users)

ankle_train_global_norm,ankle_val_global_norm,ankle_test_global_norm = \
    ng.global_normalize(ankle_train,ankle_val,ankle_test)

local_norm_data,ankle_images_local, ankle_labels_local = gi.generate_images(ankle_train,window_size,True)



