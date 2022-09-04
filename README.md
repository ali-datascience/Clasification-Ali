# Laporan Proyek Machine Learning - Muhamad Ali

# Project Overview
<hr>
Domain proyek yang dipilih dalam proyek machine learning ini adalah mengenai Case Clasification.

# Business Understanding
Client membutuhkan sebuah alat machine learning untuk perusahaan bisnis property dan booking hotel. Perusahaan hotel ingin mengoptimalkan customer yang melakukan pembatalan secara tiba-tiba, sehingga status website pemesanan online dapat diperbaiki dan mengetahui karakteristik pelanggan yang  melakukan pembatalan.

## Problem Statements
- Bagaimana profiling dari customer hotel mitranya, dari negara mana, bagaimana karakteristik dalam pemesanan hotel dilihat berdasarkan resort hotel dan city hotel
- Bagaimana karakteristik yang melakukan pembatalan sebelumnya. Hal ini ingin mengoptimalkan dengan menerapkan kebijakan baru agar tidak terjadi pembatalan yang berlebih, karena dapat merugikan perusahaan.
- Ingin memprediksi kemungkinan customer melakukan pembatalan atau tidak

## Goals

-  Mengerahui profiling dari customer berdasarakan tipe hotel (City dan resort hotel)
-  Mengetahui karakteristik dari customer yang melakukan pembatalan
-  Membuat Mechine learning untuk memprediksi kemungkinan customer melakukan pembatalan atau tidak

## Data Understanding
Dataset ini berisi data pemesanan sebuah hotel di daerah kota dan sebuah hotel di daerah resor; data yang dikoleksi seperti kapan pemesanan dilakukan, lama tinggal tamu di hotel, banyak tamu dewasa, banyak tamu anak-anak, banyak tamu bayi, jumlah ruang parkir tersedia, dan 25 fitur lainnya. Dataset ini berasal dari sebuah artikel yang diterbitkan oleh ScienceDirect. https://rpubs.com/gustavothiodorus/visualisasi_pemesanan_kamar_hotel

- variabel dalam dataset ini adalah:
- hotel: Hotel (H1 = Resort Hotel or H2 = City Hotel)
- is_canceled: Value indicating if the booking was canceled (1) or not (0)
- lead_time: Number of days that elapsed between the entering date of the booking into the PMS (Property Management System) and the arrival date
- arrival_date_year: Year of arrival date
- arrival_date_month: Month of arrival date
- arrival_date_week_number: Week number of year for arrival date
- arrival_date_day_of_month: Day of arrival date
- stays_in_weekend_nights: Number of weekend nights (Saturday or Sunday) the guest stayed or booked to stay at the hotel
- stays_in_week_nights: Number of week nights (Monday to Friday) the guest stayed or booked to stay at the hotel
- adults: Number of adults
- children: Number of children
- babies: Number of babies
- meal: Type of meal booked. Categories are presented in standard hospitality meal packages:
- Undefined/SC – no meal package
- BB – Bed & Breakfast
- HB – Half board (breakfast and one other meal – usually dinner)
- FB – Full board (breakfast, lunch and dinner)
- country: Country of origin. Categories are represented in the ISO 3155–3:2013 format
- market_segment: Market segment designation. In categories, the term “TA” means “Travel Agents” and “TO” means “Tour Operators”
- distribution_channel: Booking distribution channel. The term “TA” means “Travel Agents” and “TO” means “Tour Operators
- d: Value indicating if the booking name was from a repeated guest (1) or not (0)
- previous_cancellations: Number of previous bookings that were cancelled by the customer prior to the current booking
- previous_bookings_not_canceled: Number of previous bookings not cancelled by the customer prior to the current booking
- reserved_room_type: Code of room type reserved. Code is presented instead of designation for anonymity reasons
- assigned_room_type: Code for the type of room assigned to the booking. Sometimes the assigned room type differs from the reserved room type due to hotel operation - - reasons (e.g. overbooking) or by customer request. Code is presented instead of designation for anonymity reasons
- booking_changes: Number of changes/amendments made to the booking from the moment the booking was entered on the PMS until the moment of check-in or cancellation
- deposit_type: Indication on if the customer made a deposit to guarantee the booking. This variable can assume three categories:
- No Deposit – no deposit was made
- Non Refund – a deposit was made in the value of the total stay cost
- Refundable – a deposit was made with a value under the total cost of stay
- agent: ID of the travel agency that made the booking
- company: ID of the company/entity that made the booking or responsible for paying the booking. ID is presented instead of designation for anonymity reasons
- days_in_waiting_list: Number of days the booking was in the waiting list before it was confirmed to the customer
- customer_type: Type of booking, assuming one of four categories:
- Contract - when the booking has an allotment or other type of contract associated to it
- Group – when the booking is associated to a group
- Transient – when the booking is not part of a group or contract, and is not associated to other transient booking
- Transient-party – when the booking is transient, but is associated to at least other transient booking
- adr: Average Daily Rate as defined by dividing the sum of all lodging transactions by the total number of staying nights
- required_car_parking_spaces: Number of car parking spaces required by the customer
- total_of_special_requests: Number of special requests made by the customer (e.g. twin bed or high floor)
- reservation_status: Reservation last status, assuming one of three categories:
- Canceled – booking was canceled by the customer
- Check-Out – customer has checked in but already departed
- No-Show – customer did not check-in and did inform the hotel of the reason why
- reservation_status_date: Date at which the last status was set. This variable can be used in conjunction with the ReservationStatus to understand when was the booking canceled or when did the customer checked-out of the hotel


## Data Preparation
Sebelum membuat modeling dilakukan data preparation sebagai berikut

### Handling Null Value
 - Cek apakah data mengandung null value

df.isnull().sum()
 
- Dilakukan dropna pada kolom anak2

df.dropna(subset=['anak_anak'],inplace=True)

- Pada company diisi 0 jika customer tidak memiliki company, bisa jadi customer melakukan pembelian mandiri

df["company"].fillna("0",inplace=True)

- Dilakukan Dropna juga pada colom negara

df.dropna(subset=['negara'],inplace=True)

### Change Data Kategori dan Cek Korelasi Data
- Ubah kedalam bentuk numerik
- Convertin the predictor variable in a binary numeric variable

df['pembatalan_cat'] = df['pembatalan']
df['pembatalan_cat'].replace(to_replace='Ya', value=1, inplace=True)
df['pembatalan_cat'].replace(to_replace='Tidak',  value=0, inplace=True)
kategori = df[["tipe_hotel","meal","negara","market_segment","tipe_ruang","tipe_kamar_ditentukan","tipe_deposit","tipe_customer"]]

- **Encdoding kategori yang sudah ditentuakn dengan labelencoder**

encoded_data = LabelEncoder()
for feature in kategori:
        if feature in df.columns.values:
            df[feature+"_cat"] = encoded_data.fit_transform(df[feature])

- Rubah tipe data dengan format yang sesua (Average Daily Rate)
df['adr'] = df['adr'].str.replace(',','')
df['adr'] = df['adr'].astype(int)

- **ID company sebaiknya tidak object**

df['company'] = df['company'].astype(float)

- **Lakukan konversi nama bulan kedalam numeric**

df['bulan_kedatangan_cat'] = df['bulan_kedatangan']
df['bulan_kedatangan_cat'].replace(to_replace='January', value=1, inplace=True)
df['bulan_kedatangan_cat'].replace(to_replace='February', value=2, inplace=True)
df['bulan_kedatangan_cat'].replace(to_replace='March', value=3, inplace=True)
df['bulan_kedatangan_cat'].replace(to_replace='April', value=4, inplace=True)
df['bulan_kedatangan_cat'].replace(to_replace='May', value=5, inplace=True)
df['bulan_kedatangan_cat'].replace(to_replace='June', value=6, inplace=True)
df['bulan_kedatangan_cat'].replace(to_replace='July', value=7, inplace=True)
df['bulan_kedatangan_cat'].replace(to_replace='August', value=8, inplace=True)
df['bulan_kedatangan_cat'].replace(to_replace='September', value=9, inplace=True)
df['bulan_kedatangan_cat'].replace(to_replace='October', value=10, inplace=True)
df['bulan_kedatangan_cat'].replace(to_replace='November', value=11, inplace=True)
df['bulan_kedatangan_cat'].replace(to_replace='December', value=12, inplace=True)

- **Kemudian cek korelasi untuk menentukan fiture yang akan digunakan **

sns.heatmap(df.corr(),linewidth=.5,annot=True,cmap="RdYlGn")
fig = plt.gcf()
fig.set_size_inches(15,8)
plt.show()


- Cek Urutan korelasi terendah ke tertinggi
korelasi = df.corr()["pembatalan_cat"].sort_values()
korelasi


![image](https://user-images.githubusercontent.com/84785795/188298118-ab84e5f2-b1f8-43c1-91e6-e5a0ed6fa16a.png)

- **Berdasarkan matrix didapat beberapa variabel yang memiliki korelasi besar yaitu :**

anak_anak 0.005048
minggu_kedatangan 0.008148
tahun_kedatangan 0.016660
menginap_in_week_nights 0.024765
days_in_waiting_list 0.054186
market_segment 0.059338
dewasa 0.060017
pembatalan_sebelumnya 0.110133
negara 0.264223
waktu_tunggu 0.293123
tipe_deposit 0.468634
pembatalan 1.000000

-** Kita lihat secara spesifik matrix korelasi dari variabel2 tersebut**

cekspesifikmatrix = df[["pembatalan_cat","anak_anak","minggu_kedatangan","tahun_kedatangan","menginap_in_week_nights","days_in_waiting_list","market_segment_cat","dewasa","pembatalan_sebelumnya","negara_cat","waktu_tunggu","tipe_deposit_cat"]]

![image](https://user-images.githubusercontent.com/84785795/188298224-e71661a7-6869-4e67-92d0-8586ad30dea6.png)



## Modeling
Pada tahap modeling dipilih dilipih beberapa fiture yang dirasa memiliki korelaso baik/positive untuk dijadikan variabel input
features = df[["pembatalan_cat","tipe_deposit_cat","waktu_tunggu","negara_cat","pembatalan_sebelumnya","days_in_waiting_list","minggu_kedatangan"]]

- **Dilakukan scaling data **

scaler = MinMaxScaler()
scaled = scaler.fit_transform(features)
data_scaled = pd.DataFrame(scaled,columns=['pembatalan_cat','tipe_deposit_cat','waktu_tunggu','negara_cat','pembatalan_sebelumnya','days_in_waiting_list','minggu_kedatangan'])
data_scaled

- **Membagi data train dan test sebanyak 20 dan 80 persen**

X = data_scaled.drop('pembatalan_cat', axis=1)
y = data_scaled['pembatalan_cat']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

- **Memilih Model Logistic Regresion**

LR = LogisticRegression()
LR.fit(X_train, y_train)

y_pred = LR.predict(X_test)
logreg_test = pd.merge(X_test, y_test, left_index=True, right_index=True, how='outer')
logreg_test['prediction'] = y_pred
logreg_test


Dari hasil model tersebut mendapatkab akurasi sebesar 75%
![image](https://user-images.githubusercontent.com/84785795/188298377-e0280e22-8d73-4839-8b50-80de02402259.png)

![image](https://user-images.githubusercontent.com/84785795/188298396-03726e8c-777e-409a-96c0-d8608bb42254.png)

Recall msekitar 39% dg presisi 91% dan akurasi 76%, 

![image](https://user-images.githubusercontent.com/84785795/188298416-6b2ac635-2108-4aff-8039-4aea19ac7d45.png

Dari test AUC dapat dilihat ada di angka 0.76 ini menunjukan model kita cukup baik, namun kita akan coba denga Tunning Model untuk mendapatkan akurasi yang maksimal

- Dari hasil tunning dengan Gridsearch didapatkan Best parameters: {'C': 100, 'class_weight': None} Best cross-validation score: 0.76 dengan menerapkan scroring didapat atrix sebagai berikut

Accuracy scores:  [0.7592305697215858]
f1 scores:  [0.5480983385244317]
Precision scores:  [0.9044722486725687]
Recall scores:  [0.39318935988748505]

Akurasi model berada di sekitar 75%

- Mencoba Menerapkan Stratified k-fold cross validation dan didapatkan akurasi maximala di angka 76%


## Evaluation
Matrix evaluasi menggunakan pengukuran clasification report dan didapatkan akurasi model sekitar 76 persen dengan presisi 88 dan recal 40, sehingga asumsi model tersebut cukup baik digunakan sebagai model prediksi, evaluasi juga dilakuakan dengan membadingkan akurasi dari model lain namun didapatkan akurasi terbaik menggunakan logisti resgression meskipun secara angka hampir sama


![image](https://user-images.githubusercontent.com/84785795/188298678-780f51c1-f55a-4f0b-9231-7691ac8bf257.png)


![image](https://user-images.githubusercontent.com/84785795/188298732-b4143eab-b0d9-47d7-81aa-b255bf3ed152.png)


Tidak Terjadi peningkatan score ketika dilakukan tuning parameter, artinya dengan feature yang ada didapat score maksimal 76% apabila kita menggunakan logistik regresion. Kita akan menambahkan feature lain yaitu company dengan pertimbangan akan mempengaruhi karena bisa jadi customer ada yang memesan melalui biro jasa ataupun pesan langsung tanpa biro jasa, wich is proses cancel order akan lebih mudah dilakukan apabila customer melakukan pesanan secara langsung. Kita juga akan menambahkan feature tamu berulang, feature ini penting mengingat bisa jadi angka cancel order untuk tamu berulang/yang sudah langganan sangat kecil kemungkinan mereka melakukan batal pesanan. Feature lain yang akan ditambahkan yaitu market_segment,tipe_hotel,tipe_customer,tipe_ruang dan tipe_kamar_ditentukan karena di EDA sendiri cenderung mempengaruhi status order customer

Setelah penambahan fiture ternyata akurasinya hampir sama, namun secara bisnis feature2 baru tersebut dibutuhkan. oleh sebab itu kita akan tetap menggunakan modeling kedua dengan beberapa tambahan feature untuk memprediksi data karena secara akurasi tidak terlalu jauh, dimana pada model pertama yaitu :

Accuracy of logistic regression classifier train data: 0.761 Accuracy of logistic regression classifier test data: 0.758

perbedaanya hanya beda satu angka diblakang koma, kita ambil kesimpulan secara akurasi dan best accuration model logistik ini memang berada di kiradan 76 persen

  
 
## KESIMPULAN/SARAN

Perusahaan perlu membuat promosi untuk resort hotel karena presentasi order yang lumayan kecil jika dibandingkan dengan city hotel, kemudian matrix yang mempengaruhi angka pembatalan pemesanan juga perlu di reduse seperti misalkan mengharuskan untuk deposite terlebih dahulu untuk pemesanan hotel dan perlu melakukan mekanisme promosi pada negara-negara dengan angka order terkecil, mungkin bisa mengadayakan upaya promosi kerjasama dengan destinasi wisata setempat untuk melakukan upaya marketing campaign atau promosi lainya dengan target negara-negara tersebut. Bisa juga mengadakan sistem loyalti point/rewards ataupun referal bagi pengunjung setia hotel.
