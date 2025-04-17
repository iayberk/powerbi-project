# Gerekli kütüphaneleri içe aktar
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Excel dosyasını yükle (dosya yolunu kendine göre değiştir!)
file_path = "C:\\Users\\ayber\\Downloads\\Retail-Supply-Chain-Sales-Dataset.xlsx"
df = pd.read_excel(file_path, sheet_name='Retails Order Full Dataset')

# Tarih sütununu datetime formatına çevir
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Yıl bazında toplam satış hesapla
annual_sales = df.groupby(df['Order Date'].dt.year)['Sales'].sum().reset_index()
annual_sales.columns = ['Year', 'Total_Sales']

# Sonuçları yazdır
print("Yıllara Göre Satış Hacmi:")
print(annual_sales)

# Veri: Yıl ve Satış
X = annual_sales['Year'].values.reshape(-1, 1)
y = annual_sales['Total_Sales'].values

# Doğrusal Regresyon Modeli Eğit
model = LinearRegression()
model.fit(X, y)

# 2014-2018 arası tahmin yap
future_years = np.array([2014, 2015, 2016, 2017, 2018]).reshape(-1, 1)
predicted_sales = model.predict(future_years)

# Grafik çiz
plt.figure(figsize=(8, 5))
plt.plot(annual_sales['Year'], annual_sales['Total_Sales'], marker='o', label='Gerçek Satış')
plt.plot(future_years.flatten(), predicted_sales, linestyle='--', color='orange', label='Tahmin Edilen Satış (Trend)')

# 2018 tahminini kırmızı nokta olarak işaretle
plt.scatter(2018, predicted_sales[-1], color='red', s=100, zorder=5, label=f'2018 Tahmin: {predicted_sales[-1]:,.2f}')

# Grafik başlık ve etiketler
plt.title('Yıllara Göre Satış Hacmi ve 2018 Tahmini')
plt.xlabel('Yıl')
plt.ylabel('Toplam Satış')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Grafik göster
plt.show()

# 2018 tahmini sonucu yazdır
print(f"Modelin 2018 için tahmini satış hacmi: {predicted_sales[-1]:,.2f}")
