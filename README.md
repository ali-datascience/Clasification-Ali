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
Sebelum membuat modeling dilakukan data preparation sebagai berikut,

- Dari hasil checking ditemukan ada data yang null, sehingga dilakukan Handling Null Value
- Dilakukan dropna pada kolom anak2, 
- Pada company diisi 0 jika customer tidak memiliki company, bisa jadi customer melakukan pembelian mandiri sehingga tidak perlu dilakukan drop missing value
- Pada kolom negara dilakukan handling missing value degangan cara dropna, karena asumsi setiap customer pasti berasal dari sebuah negara atau memiliki negara asal, jika tidak ada negaranya maka tidak bisa dilakukan profiling customer
- Pada column pembatalan di Convert di convert ke numeric agar bisa diproses dalam model mechine learning
- Pada kolom yang lain dilakukan proses Encdoding kategori yang sudah ditentuakn dengan labelencoder**
- Rubah tipe data dengan format yang sesuai untuk (Average Daily Rate), karena tipe data yang ada masih object
- ID company sebaiknya tidak object sehingga dalam hal ini perlu dilakukan perubahan tipe data
- konversi nama bulan kedalam numeric dari bulan 1 sampai 12
- Kemudian cek korelasi untuk menentukan fiture yang akan digunakan

![image](https://user-images.githubusercontent.com/84785795/188298118-ab84e5f2-b1f8-43c1-91e6-e5a0ed6fa16a.png)

**Berdasarkan matrix didapat beberapa variabel yang memiliki korelasi besar yaitu :**

- anak_anak 0.005048
- minggu_kedatangan 0.008148
- tahun_kedatangan 0.016660
- menginap_in_week_nights 0.024765
- days_in_waiting_list 0.054186
- market_segment 0.059338
- dewasa 0.060017
- pembatalan_sebelumnya 0.110133
- negara 0.264223
- waktu_tunggu 0.293123
- tipe_deposit 0.468634
- pembatalan 1.000000

-** Jika Kita lihat secara spesifik matrix korelasi dari variabel2 tersebut adalah sebagi berikut**

![image](https://user-images.githubusercontent.com/84785795/188298224-e71661a7-6869-4e67-92d0-8586ad30dea6.png)



## Modeling

Pada tahap modeling dipilih dilipih beberapa fiture yang dirasa memiliki korelasi baik/positive untuk dijadikan variabel input yaitu :

"pembatalan_cat","tipe_deposit_cat","waktu_tunggu","negara_cat","pembatalan_sebelumnya","days_in_waiting_list","minggu_kedatangan"

- Feture yang dipilih dilakukan scaling data menggunakan MinMaxScaler pada variabel input agar range data tidak terlalu berbeda dan melebar
- Data dibagi data train dan test sebanyak 20 dan 80 persen
- Model yang dipilih yaitu Logistic Regression dengan pertimbangan memiliki akurasi yang lebih baik dari hasil uji coba dengan model lainya

Dari hasil model tersebut mendapatkan akurasi sebesar 75%

![image](https://user-images.githubusercontent.com/84785795/188315517-2597a791-67ee-4ff9-b7b2-f1f05f257af4.png)

![image](https://user-images.githubusercontent.com/84785795/188315529-7bd60bf8-ad52-4eb5-807c-3253247caedf.png)

![image](https://user-images.githubusercontent.com/84785795/188315550-78f474f4-27fc-4270-a1a9-be22e186b8e5.png)


Recall msekitar 39% dg presisi 91% dan akurasi 76%, 

![image](https://user-images.githubusercontent.com/84785795/188315581-04fb28fb-68b5-4f00-bcea-158b6191727c.png)


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


![image](https://user-images.githubusercontent.com/84785795/188315607-1a5c1c25-c130-4bf1-8827-ca9cc43c80ce.png)



Tidak Terjadi peningkatan score ketika dilakukan tuning parameter, artinya dengan feature yang ada didapat score maksimal 76% apabila kita menggunakan logistik regresion. Kita akan menambahkan feature lain yaitu company dengan pertimbangan akan mempengaruhi karena bisa jadi customer ada yang memesan melalui biro jasa ataupun pesan langsung tanpa biro jasa, wich is proses cancel order akan lebih mudah dilakukan apabila customer melakukan pesanan secara langsung. Kita juga akan menambahkan feature tamu berulang, feature ini penting mengingat bisa jadi angka cancel order untuk tamu berulang/yang sudah langganan sangat kecil kemungkinan mereka melakukan batal pesanan. Feature lain yang akan ditambahkan yaitu market_segment,tipe_hotel,tipe_customer,tipe_ruang dan tipe_kamar_ditentukan karena di EDA sendiri cenderung mempengaruhi status order customer

Setelah penambahan fiture ternyata akurasinya hampir sama, namun secara bisnis feature2 baru tersebut dibutuhkan. oleh sebab itu kita akan tetap menggunakan modeling kedua dengan beberapa tambahan feature untuk memprediksi data karena secara akurasi tidak terlalu jauh, dimana pada model pertama yaitu :

Accuracy of logistic regression classifier train data: 0.761 Accuracy of logistic regression classifier test data: 0.758

perbedaanya hanya beda satu angka diblakang koma, kita ambil kesimpulan secara akurasi dan best accuration model logistik ini memang berada di kiradan 76 persen

  
 
## KESIMPULAN/SARAN

**profiling dari customer berdasarakan tipe hotel (City dan resort hotel)**

Berdasarkan data  5 negara dengan customer terbanyak yaitu Portugal sebanyak 21071, disusul dengan 4 negara lain yaitu

United Kingdom 9676
France 8481
Spain 6391
Germany 6069
dan Jumlah Negara Dengan Customer paling sedikit yaitu sebanyak 31 negara

![image](https://user-images.githubusercontent.com/84785795/188299550-53fde150-44b9-48b5-9953-e913eaa001f8.png)

*Dari pie chart dapat dilihat customer lebih banyak memesan City Hotel dibandingkan dengan Resort Hotel, dengan perbandungan 61.6 banding 38.5 persen *

![image](https://user-images.githubusercontent.com/84785795/188299531-9d18f486-83d0-444f-8fcc-d96111bb5850.png)


Kital lihat grafik pembatalan berdasarkan tipe hotel

![image](https://user-images.githubusercontent.com/84785795/188299607-0f372a6a-7ab0-4aee-a0e0-a869e19bc336.png)

City hotel memiliki jumlah pembatalan paling banyak sekitar 74.9%, berbanding lurus dengan jumlah pesanan terbanyak yaitu dari city hotel. Berarti dalam hal ini perusahaan perlu meningkatkan upaya promosi agar customer juga dapat tertarik pada tipe Resort Hotel mengingat angka presentasi pemesanan Resort Hotel yang kecil. selain itu juga perlu dicari tau mengapa customer banyak melakukan pembatalan pesanan pada city hotel

**Karakteristik dari customer yang melakukan pembatalan sebelumnya**

![image](https://user-images.githubusercontent.com/84785795/188299385-67835d7e-4d46-4bae-b316-c28db71ff309.png)

![image](https://user-images.githubusercontent.com/84785795/188299395-0fde451a-4617-4f72-976e-8bdb92eea483.png)

![image](https://user-images.githubusercontent.com/84785795/188299404-f22242c5-b103-425f-a287-04f8e04b9003.png)

![image](https://user-images.githubusercontent.com/84785795/188299408-585aa52e-ab02-4bc7-af83-348c3071df19.png)

![image](https://user-images.githubusercontent.com/84785795/188299416-4bcd25ae-e5fd-4db5-97b2-d8835e6f5f4b.png)

![image](https://user-images.githubusercontent.com/84785795/188299422-dc37e6f7-f5bf-4e26-bfa9-5d0f608405bd.png)

![image](https://user-images.githubusercontent.com/84785795/188299426-468cab23-814f-4363-b5ce-4eca1ccad408.png)

![image](https://user-images.githubusercontent.com/84785795/188299433-4fc320ce-aea2-4d2c-b90c-3697b3147026.png)

![image](https://user-images.githubusercontent.com/84785795/188299436-6658d1cb-ede8-4060-95f1-d31fd445fd0d.png)

![image](https://user-images.githubusercontent.com/84785795/188299450-ec342aa3-d824-4861-a00f-6316260b264f.png)

![image](https://user-images.githubusercontent.com/84785795/188299453-e0b4564e-6888-4814-8834-35233956520f.png)

![image](https://user-images.githubusercontent.com/84785795/188299457-657633b0-c4d3-45a4-acf0-8f8257ba5742.png)



- Ada sebenyak 6484 customer yang sudah pernah melakukan pembatalan sebelumnya
- Jumlah pemesanan dengan 2 orang dewasa dan tidak memiliki anak-anak dan babies cenderung lebih banyak melakukan pembatalan sebelumnya.
- Tipe meal dengan tipe "BB" juga cenderung banyak melakukan pembatalan.
- Market segment paling banyak yang melakukan pembatalan sebelumnya yaitu dari Offile TA/TO dan Group.
- Customer baru / bukan tamu berulang lebih banyak dalam melakukan pembatalan sebelumnya.
- Untuk tipe ruang dan kamar yang ditentukan yang paling banyak melakukan pembatalan sebelumnya adalah tipe A.
- Customer yang malakukan pembatalan sebelumnya lebih banyak dari customer yang tidak melakukan perubahan pemesanan
- Jenis Deposit dengan pilihan tanpa deposito dan Non Refund lebih banyak dalam melakukan pematalan
- Tipe customer transien dan tidak membutuhkan parkir paling banyak melakukan pembatalan


**SARAN**

Perusahaan perlu membuat promosi untuk resort hotel karena presentasi order yang lumayan kecil jika dibandingkan dengan city hotel, kemudian matrix yang mempengaruhi angka pembatalan pemesanan juga perlu di reduse seperti misalkan mengharuskan untuk deposite terlebih dahulu untuk pemesanan hotel dan perlu melakukan mekanisme promosi pada negara-negara dengan angka order terkecil, mungkin bisa mengadayakan upaya promosi kerjasama dengan destinasi wisata setempat untuk melakukan upaya marketing campaign atau promosi lainya dengan target negara-negara tersebut. Bisa juga mengadakan sistem loyalti point/rewards ataupun referal bagi pengunjung setia hotel.
