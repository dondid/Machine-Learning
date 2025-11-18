import json
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean, stdev

# ============================================================================
# EXERCIȚIUL 1: Crearea și analiza datelor
# ============================================================================

# Varianta A: Date SUV
suv_data = [
    {
        "type": "Toyota RAV4",
        "engine": "2.5L",
        "fuel": "Hybrid",
        "year": 2021,
        "mileage": 35000,
        "price": 28500
    },
    {
        "type": "Honda CR-V",
        "engine": "1.5L Turbo",
        "fuel": "Petrol",
        "year": 2020,
        "mileage": 48000,
        "price": 24800
    },
    {
        "type": "Mazda CX-5",
        "engine": "2.0L",
        "fuel": "Petrol",
        "year": 2022,
        "mileage": 22000,
        "price": 31200
    },
    {
        "type": "Ford Escape",
        "engine": "2.0L",
        "fuel": "Hybrid",
        "year": 2019,
        "mileage": 67000,
        "price": 21500
    },
    {
        "type": "Nissan Rogue",
        "engine": "2.5L",
        "fuel": "Petrol",
        "year": 2021,
        "mileage": 41000,
        "price": 26300
    },
    {
        "type": "Subaru Forester",
        "engine": "2.5L",
        "fuel": "Petrol",
        "year": 2020,
        "mileage": 53000,
        "price": 23900
    }
]

# Varianta B: Date Apartamente
apartment_data = [
    {
        "location": "Centru",
        "rooms": 3,
        "surface": 85,
        "floor": 4,
        "bathrooms": 2,
        "distance_to_transport": 200,
        "price": 125000
    },
    {
        "location": "Rahova",
        "rooms": 2,
        "surface": 58,
        "floor": 2,
        "bathrooms": 1,
        "distance_to_transport": 350,
        "price": 78000
    },
    {
        "location": "Brazda",
        "rooms": 4,
        "surface": 110,
        "floor": 1,
        "bathrooms": 2,
        "distance_to_transport": 500,
        "price": 95000
    },
    {
        "location": "Centru",
        "rooms": 2,
        "surface": 62,
        "floor": 7,
        "bathrooms": 1,
        "distance_to_transport": 150,
        "price": 98000
    },
    {
        "location": "1 Mai",
        "rooms": 3,
        "surface": 75,
        "floor": 3,
        "bathrooms": 1,
        "distance_to_transport": 400,
        "price": 82000
    },
    {
        "location": "Rovine",
        "rooms": 3,
        "surface": 92,
        "floor": 5,
        "bathrooms": 2,
        "distance_to_transport": 300,
        "price": 108000
    },
    {
        "location": "Craiovița",
        "rooms": 1,
        "surface": 42,
        "floor": 2,
        "bathrooms": 1,
        "distance_to_transport": 600,
        "price": 52000
    }
]

# Salvarea datelor în fișiere JSON
with open('suv_data.json', 'w', encoding='utf-8') as f:
    json.dump(suv_data, f, indent=2, ensure_ascii=False)

with open('apartment_data.json', 'w', encoding='utf-8') as f:
    json.dump(apartment_data, f, indent=2, ensure_ascii=False)

print("=" * 70)
print("EXERCIȚIUL 1: Analiza minimului și maximului de preț")
print("=" * 70)

# Analiza pentru SUV-uri
print("\n--- ANALIZA SUV-URI ---")
suv_prices = [car['price'] for car in suv_data]
min_price_suv = min(suv_prices)
max_price_suv = max(suv_prices)

min_suv = next(car for car in suv_data if car['price'] == min_price_suv)
max_suv = next(car for car in suv_data if car['price'] == max_price_suv)

print(f"\nPreț minim: €{min_price_suv:,}")
print(f"Caracteristici: {min_suv['type']}, {min_suv['engine']}, {min_suv['fuel']}, "
      f"An {min_suv['year']}, {min_suv['mileage']:,} km")

print(f"\nPreț maxim: €{max_price_suv:,}")
print(f"Caracteristici: {max_suv['type']}, {max_suv['engine']}, {max_suv['fuel']}, "
      f"An {max_suv['year']}, {max_suv['mileage']:,} km")

# Analiza pentru Apartamente
print("\n--- ANALIZA APARTAMENTE ---")
apt_prices = [apt['price'] for apt in apartment_data]
min_price_apt = min(apt_prices)
max_price_apt = max(apt_prices)

min_apt = next(apt for apt in apartment_data if apt['price'] == min_price_apt)
max_apt = next(apt for apt in apartment_data if apt['price'] == max_price_apt)

print(f"\nPreț minim: €{min_price_apt:,}")
print(f"Caracteristici: {min_apt['location']}, {min_apt['rooms']} camere, "
      f"{min_apt['surface']} m², etaj {min_apt['floor']}, "
      f"{min_apt['bathrooms']} băi, {min_apt['distance_to_transport']}m transport")

print(f"\nPreț maxim: €{max_price_apt:,}")
print(f"Caracteristici: {max_apt['location']}, {max_apt['rooms']} camere, "
      f"{max_apt['surface']} m², etaj {max_apt['floor']}, "
      f"{max_apt['bathrooms']} băi, {max_apt['distance_to_transport']}m transport")

# ============================================================================
# EXERCIȚIUL 2: Grafice scatter plot
# ============================================================================
print("\n" + "=" * 70)
print("EXERCIȚIUL 2: Crearea graficelor")
print("=" * 70)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Grafic a: Mileage vs Price pentru SUV-uri
mileages = [car['mileage'] for car in suv_data]
suv_prices_plot = [car['price'] for car in suv_data]

ax1.scatter(mileages, suv_prices_plot, s=100, alpha=0.6, color='steelblue', edgecolors='black')
ax1.set_xlabel('Kilometraj (km)', fontsize=12)
ax1.set_ylabel('Preț (€)', fontsize=12)
ax1.set_title('Kilometraj vs Preț SUV', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.ticklabel_format(style='plain', axis='both')

# Grafic b: Surface vs Price pentru Apartamente
surfaces = [apt['surface'] for apt in apartment_data]
apt_prices_plot = [apt['price'] for apt in apartment_data]

ax2.scatter(surfaces, apt_prices_plot, s=100, alpha=0.6, color='coral', edgecolors='black')
ax2.set_xlabel('Suprafață (m²)', fontsize=12)
ax2.set_ylabel('Preț (€)', fontsize=12)
ax2.set_title('Suprafață vs Preț Apartamente', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.ticklabel_format(style='plain', axis='both')

plt.tight_layout()
plt.savefig('scatter_plots.png', dpi=300, bbox_inches='tight')
print("\nGraficele scatter au fost salvate în 'scatter_plots.png'")
plt.show()

# ============================================================================
# EXERCIȚIUL 3: Verificarea pragului de preț
# ============================================================================
print("\n" + "=" * 70)
print("EXERCIȚIUL 3: Verificarea pragului de preț")
print("=" * 70)

threshold_suv = 25000
threshold_apt = 90000

above_threshold_suv = sum(1 for price in suv_prices if price > threshold_suv)
above_threshold_apt = sum(1 for price in apt_prices if price > threshold_apt)

print(f"\nSUV-uri:")
print(f"Prag: €{threshold_suv:,}")
print(f"Număr de prețuri peste prag: {above_threshold_suv} din {len(suv_prices)}")
print(f"Procent: {(above_threshold_suv/len(suv_prices)*100):.1f}%")

print(f"\nApartamente:")
print(f"Prag: €{threshold_apt:,}")
print(f"Număr de prețuri peste prag: {above_threshold_apt} din {len(apt_prices)}")
print(f"Procent: {(above_threshold_apt/len(apt_prices)*100):.1f}%")

# ============================================================================
# EXERCIȚIUL 4: Statistici și vizualizări (Boxplot și Violin Plot)
# ============================================================================
print("\n" + "=" * 70)
print("EXERCIȚIUL 4: Statistici și vizualizări")
print("=" * 70)

# Calcularea statisticilor pentru SUV-uri
mean_suv = mean(suv_prices)
std_suv = stdev(suv_prices)

print("\n--- STATISTICI SUV-URI ---")
print(f"Media prețurilor: €{mean_suv:,.2f}")
print(f"Deviația standard: €{std_suv:,.2f}")
print(f"Interval de încredere (±1σ): €{mean_suv-std_suv:,.2f} - €{mean_suv+std_suv:,.2f}")

# Calcularea statisticilor pentru Apartamente
mean_apt = mean(apt_prices)
std_apt = stdev(apt_prices)

print("\n--- STATISTICI APARTAMENTE ---")
print(f"Media prețurilor: €{mean_apt:,.2f}")
print(f"Deviația standard: €{std_apt:,.2f}")
print(f"Interval de încredere (±1σ): €{mean_apt-std_apt:,.2f} - €{mean_apt+std_apt:,.2f}")

# Crearea vizualizărilor
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Boxplot pentru SUV-uri
axes[0, 0].boxplot(suv_prices, vert=True, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5))
axes[0, 0].set_ylabel('Preț (€)', fontsize=12)
axes[0, 0].set_title('Boxplot - Prețuri SUV', fontsize=14, fontweight='bold')
axes[0, 0].set_xticklabels(['SUV-uri'])
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Violin plot pentru SUV-uri
parts = axes[0, 1].violinplot([suv_prices], vert=True, showmeans=True, showmedians=True)
for pc in parts['bodies']:
    pc.set_facecolor('lightblue')
    pc.set_alpha(0.7)
axes[0, 1].set_ylabel('Preț (€)', fontsize=12)
axes[0, 1].set_title('Violin Plot - Prețuri SUV', fontsize=14, fontweight='bold')
axes[0, 1].set_xticklabels(['', 'SUV-uri'])
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Boxplot pentru Apartamente
axes[1, 0].boxplot(apt_prices, vert=True, patch_artist=True,
                    boxprops=dict(facecolor='lightcoral', alpha=0.7),
                    medianprops=dict(color='darkred', linewidth=2),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5))
axes[1, 0].set_ylabel('Preț (€)', fontsize=12)
axes[1, 0].set_title('Boxplot - Prețuri Apartamente', fontsize=14, fontweight='bold')
axes[1, 0].set_xticklabels(['Apartamente'])
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Violin plot pentru Apartamente
parts = axes[1, 1].violinplot([apt_prices], vert=True, showmeans=True, showmedians=True)
for pc in parts['bodies']:
    pc.set_facecolor('lightcoral')
    pc.set_alpha(0.7)
axes[1, 1].set_ylabel('Preț (€)', fontsize=12)
axes[1, 1].set_title('Violin Plot - Prețuri Apartamente', fontsize=14, fontweight='bold')
axes[1, 1].set_xticklabels(['', 'Apartamente'])
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('statistical_plots.png', dpi=300, bbox_inches='tight')
print("\nGraficele statistice au fost salvate în 'statistical_plots.png'")
plt.show()

print("\n" + "=" * 70)
print("Analiza completă finalizată!")
print("=" * 70)