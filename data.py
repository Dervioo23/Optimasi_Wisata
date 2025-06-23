import requests
import json
import csv

# Fungsi untuk mengambil data tempat wisata dari Overpass API
def get_tourist_attractions():
    try:
        query = """
        [out:json];
        area[name="Padang"]->.searchArea;
        (
          node[access=customers][tourism=attraction](area.searchArea);
          node[access=customers][tourism=museum](area.searchArea);
          node[access=customers][tourism=beach](area.searchArea);
        );
        out center;
        """
        url = "https://overpass-api.de/api/interpreter"
        response = requests.post(url, data={'data': query}, timeout=10)
        
        if response.status_code != 200:
            print(f"Gagal mengambil data dari Overpass API: {response.status_code}")
            return []
    except Exception as e:
        print(f"Terjadi kesalahan saat mengambil data dari Overpass API: {e}")
        return []

    data = response.json()
    attractions = []
    for element in data['elements']:
        attraction = {
            'ID': len(attractions) + 1,
            'Nama': element['tags'].get('name', 'Unknown'),
            'Latitude': element['lat'],
            'Longitude': element['lon']
        }
        attractions.append(attraction)
    
    # Jika data dari API kosong, gunakan data dummy
    if not attractions:
        print("Tidak ada data dari Overpass API, menggunakan data dummy.")
        attractions = [
            {'ID': 1, 'Nama': 'Pantai Air Manis', 'Latitude': -0.9551, 'Longitude': 100.3572},
            {'ID': 2, 'Nama': 'Jembatan Siti Nurbaya', 'Latitude': -0.9478, 'Longitude': 100.3635},
            {'ID': 3, 'Nama': 'Museum Adityawarman', 'Latitude': -0.9486, 'Longitude': 100.3698},
            {'ID': 4, 'Nama': 'Bukit Gado-Gado', 'Latitude': -0.9392, 'Longitude': 100.3517},
            {'ID': 5, 'Nama': 'Pantai Padang', 'Latitude': -0.9512, 'Longitude': 100.3589}
        ]
    
    return attractions

# Fungsi untuk menghitung matriks jarak menggunakan OpenRouteService API
def get_distance_matrix(attractions, api_key):
    locations = [[attr['Longitude'], attr['Latitude']] for attr in attractions]
    payload = {
        "locations": locations,
        "metrics": ["distance"],
        "units": "km"
    }
    headers = {'Authorization': api_key}
    url = 'https://api.openrouteservice.org/v2/matrix/driving-car'
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code != 200:
        print(f"Gagal mengambil data dari ORS API: {response.status_code}")
        return None
    
    data = response.json()
    distances = data['distances']  # Jarak dalam km
    return distances

# Fungsi untuk menyimpan semua data ke satu file CSV
def save_to_csv(attractions, distances):
    # Data tambahan (manual) untuk durasi, rating, dan jam buka/tutup
    additional_data = [
        {'ID': 1, 'Durasi Kunjungan (menit)': 60, 'Rating': 4.5, 'Jam Buka': '06:00', 'Jam Tutup': '18:00'},
        {'ID': 2, 'Durasi Kunjungan (menit)': 30, 'Rating': 4.0, 'Jam Buka': '00:00', 'Jam Tutup': '23:59'},
        {'ID': 3, 'Durasi Kunjungan (menit)': 90, 'Rating': 4.2, 'Jam Buka': '08:00', 'Jam Tutup': '16:00'},
        {'ID': 4, 'Durasi Kunjungan (menit)': 45, 'Rating': 3.8, 'Jam Buka': '06:00', 'Jam Tutup': '18:00'},
        {'ID': 5, 'Durasi Kunjungan (menit)': 60, 'Rating': 4.3, 'Jam Buka': '06:00', 'Jam Tutup': '20:00'}
    ]
    
    # Gabungkan data tambahan dengan data tempat wisata
    for attr in attractions:
        for add_data in additional_data:
            if add_data['ID'] == attr['ID']:
                attr.update(add_data)
    
    # Siapkan header dan data untuk CSV
    attraction_names = [attr['Nama'] for attr in attractions]
    fieldnames = ['ID', 'Nama', 'Latitude', 'Longitude', 'Durasi Kunjungan (menit)', 'Rating', 'Jam Buka', 'Jam Tutup']
    fieldnames.extend([f"Jarak ke {name} (km)" for name in attraction_names])
    
    # Siapkan baris data
    rows = []
    for i, attr in enumerate(attractions):
        row = attr.copy()
        # Tambahkan jarak ke lokasi lain
        if distances is not None:
            for j, distance in enumerate(distances[i]):
                row[f"Jarak ke {attraction_names[j]} (km)"] = f"{distance:.1f}" if distance is not None else "N/A"
        else:
            # Gunakan data dummy jika ORS API gagal
            dummy_distances = [
                [0.0, 2.5, 3.0, 4.2, 1.8],
                [2.5, 0.0, 1.2, 3.5, 1.0],
                [3.0, 1.2, 0.0, 2.8, 1.5],
                [4.2, 3.5, 2.8, 0.0, 3.0],
                [1.8, 1.0, 1.5, 3.0, 0.0]
            ]
            for j, distance in enumerate(dummy_distances[i]):
                row[f"Jarak ke {attraction_names[j]} (km)"] = f"{distance:.1f}"
        rows.append(row)
    
    # Simpan ke CSV
    with open('tourist_data_padang.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("Data disimpan ke 'tourist_data_padang.csv'")

def main():
    # Ganti dengan kunci API Anda
    ors_api_key = '5b3ce3597851110001cf624884cc86f053634c8286e368634ea64118'
    
    # Langkah 1: Ambil data tempat wisata
    print("Mengambil data tempat wisata dari Overpass API...")
    attractions = get_tourist_attractions()
    
    if not attractions:
        print("Tidak ada data tempat wisata yang ditemukan.")
        return
    
    # Langkah 2: Hitung matriks jarak
    print("Menghitung matriks jarak menggunakan OpenRouteService API...")
    distances = get_distance_matrix(attractions, ors_api_key)
    
    # Langkah 3: Simpan semua data ke satu file CSV
    save_to_csv(attractions, distances)

if __name__ == "__main__":
    main()