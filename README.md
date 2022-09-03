# Laporan Proyek Machine Learning - Muhamad Ali

# Project Overview
<hr>
Domain proyek yang dipilih dalam proyek machine learning ini adalah mengenai Case Clasification.

# Business Understanding
Client membutuhkan sebuah alat machine learning untuk perusahaan bisnis property dan booking hotel. Perusahaan hotel ingin mengoptimalkan customer yang melakukan pembatalan secara tiba-tiba, sehingga status website pemesanan online dapat diperbaiki dan mengetahui pelanggan yang akan melakukan pembatalan.

## Problem Statements
- Perusahaan ingin mengetahui bagaimana profiling dari customer hotel mitranya, dari negara mana, bagaimana karakteristik dalam pemesanan hotel dilihat berdasarkan resort hotel dan city hotel
- Perusahaan ingin fokus pada karakteristik yang melakukan pembatalan sebelumnya. Hal ini ingin mengoptimalkan dengan menerapkan kebijakan baru agar tidak terjadi pembatalan yang berlebih, karena dapat merugikan perusahaan.
- Membuat mechine learning dengan feature-feature dan berikan kebijakan berdasarkan model mechine learning yang kamu buat untuk mengoptimalkan website pemesanan online hotel

## Solusi

- Untuk mencari tau profiling customer maka dilakukan filter customer yang benar2 melakukan pemesanan (tidak melakukan pembatalan)
-  Melakukan EDA dan Visualisai untuk mencari tau karakteristik customer
-  Membuat Logistic Regression

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

## Modeling
Menggunakan Logistic Regression Dg Akurasi sekitar 76%. Perbindingan dengan model lain namun akurasi maximal berada di angka yang sama
Sudah dilakukan Tunning namun akurasi tidak terlalu signifikan

LR = LogisticRegression()
LR.fit(X_train, y_train)

y_pred = LR.predict(X_test)
logreg_test = pd.merge(X_test, y_test, left_index=True, right_index=True, how='outer')
logreg_test['prediction'] = y_pred
logreg_test


Feature yang ditambahkan adalah :
 "pembatalan","tipe_deposit","waktu_tunggu","negara","pembatalan_sebelumnya","days_in_waiting_list","minggu_kedatangan", "company","tamu_berulang","tipe_hotel" 
![image](https://user-images.githubusercontent.com/84785795/188253778-0869a0a8-9c55-4f18-834b-2603d1ed6ecc.png)



## Evaluation
Matrix evaluasi menggunakan pengukuran clasification report dan didapatkan akurasi model sekitar 76 persen dengan presisi 88 dan recal 40, sehingga asumsi model tersebut cukup baik digunakan sebagai model prediksi

print(metrics.classification_report(y_test, y_pred))

  
 

