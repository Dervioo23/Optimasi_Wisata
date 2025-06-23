import streamlit as st
import pandas as pd
import numpy as np
import random
import math
import folium
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
import osmnx as ox
import geopandas as gpd
import networkx as nx
from datetime import datetime, timedelta

# --- FUNGSI PENGAMBILAN & PEMROSESAN DATA ---

@st.cache_data
def fetch_osm_data():
    """Mengambil data POI pariwisata dari OpenStreetMap untuk kota Padang."""
    st.info("Memulai pengambilan data dari OpenStreetMap. Proses ini mungkin memakan waktu beberapa saat...")
    padang_boundary = ox.geocode_to_gdf("Padang, Sumatera Barat, Indonesia")
    tags = {'tourism': True}
    gdf = ox.features_from_polygon(padang_boundary.geometry.iloc[0], tags)
    
    # Proses data
    df = gdf[['name', 'tourism', 'geometry']].dropna(subset=['name'])
    df['latitude'] = df.geometry.centroid.y
    df['longitude'] = df.geometry.centroid.x
    
    # Estimasi durasi berdasarkan jenis tempat
    duration_map = {
        'museum': 60, 'gallery': 45, 'attraction': 30, 'artwork': 20,
        'viewpoint': 25, 'theme_park': 120, 'zoo': 90
    }
    df['Estimasi Durasi Kunjungan (menit)'] = df['tourism'].map(duration_map).fillna(20)
    df['Jenis Tempat'] = df['tourism']
    
    df = df.reset_index(drop=True)
    df.index += 1
    df.index.name = 'No'
    
    # Simpan ke CSV
    df_out = df[['name', 'latitude', 'longitude', 'Estimasi Durasi Kunjungan (menit)', 'Jenis Tempat']].rename(
        columns={'name': 'Nama Tempat Wisata'}
    )
    df_out.to_csv("data_tempat_wisata_padang.csv")
    st.success("Data pariwisata Padang berhasil diambil dan disimpan ke `data_tempat_wisata_padang.csv`.")
    return df_out

@st.cache_data
def get_road_network(city_name="Padang, Indonesia"):
    """Mengambil dan menyimpan graf jaringan jalan dari OSM."""
    G = ox.graph_from_place(city_name, network_type='drive')
    return G

# --- FUNGSI PERHITUNGAN & ALGORITMA ---

def calculate_time_matrix(locations_df, speed_kmh):
    """Menghitung matriks waktu tempuh antar lokasi menggunakan jaringan jalan raya."""
    G = get_road_network()
    locations = list(locations_df.itertuples(index=False))
    n = len(locations)
    time_matrix = np.zeros((n, n))
    speed_mps = speed_kmh * 1000 / 3600  # Konversi km/jam ke m/s

    # Dapatkan node terdekat untuk setiap lokasi sekali saja
    nodes = [ox.nearest_nodes(G, loc.longitude, loc.latitude) for loc in locations]

    for i in range(n):
        for j in range(n):
            if i != j:
                try:
                    # Hitung panjang rute terpendek (dalam meter)
                    length = nx.shortest_path_length(G, nodes[i], nodes[j], weight='length')
                    # Hitung waktu tempuh (dalam menit)
                    time_matrix[i, j] = (length / speed_mps) / 60
                except nx.NetworkXNoPath:
                    # Fallback jika tidak ada rute (jarang terjadi di jaringan 'drive')
                    time_matrix[i, j] = np.inf
    return time_matrix

def fitness(route, time_matrix, durations):
    """
    Fitness function: menghitung total waktu (waktu tempuh + waktu kunjungan).
    Tujuan: minimalkan total waktu, sehingga fitness = 1 / total_waktu.
    """
    total_travel_time = sum(time_matrix[route[i]][route[i+1]] for i in range(len(route) - 1))
    total_visit_time = sum(durations[i] for i in route)
    total_time = total_travel_time + total_visit_time
    return 1 / total_time if total_time > 0 else 0

def selection(population, fitness_values):
    """Seleksi turnamen."""
    tournament = random.sample(list(zip(population, fitness_values)), k=3)
    tournament.sort(key=lambda x: x[1], reverse=True)
    return tournament[0][0]

def crossover(parent1, parent2):
    """Ordered Crossover (OX1)."""
    size = len(parent1)
    child = [-1] * size
    start, end = sorted(random.sample(range(size), 2))
    child[start:end] = parent1[start:end]
    
    p2_genes = [gene for gene in parent2 if gene not in child]
    
    # Isi sisa gen dari parent2
    for i in range(size):
        if child[i] == -1:
            child[i] = p2_genes.pop(0)
    return child

def mutate(route, mutation_rate):
    """Mutasi swap."""
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(route)), 2)
        route[i], route[j] = route[j], route[i]
    return route

def genetic_algorithm(locations_df, start_index, pop_size, generations, mutation_rate, elitism_size, speed_kmh):
    """Algoritma Genetika untuk mencari rute wisata optimal."""
    
    # Pisahkan titik awal dari lokasi lain yang akan dioptimalkan
    other_locations_df = locations_df.drop(start_index).reset_index(drop=True)
    location_indices = list(other_locations_df.index)
    
    # Hitung matriks waktu untuk semua lokasi terpilih
    full_time_matrix = calculate_time_matrix(locations_df, speed_kmh)
    
    # Durasi kunjungan untuk lokasi yang dioptimalkan
    durations_to_optimize = other_locations_df['Estimasi Durasi Kunjungan (menit)'].values
    
    # Buat matriks waktu hanya untuk lokasi yang dioptimalkan
    indices_to_optimize = list(other_locations_df.index)
    time_matrix_for_ga = np.array([[full_time_matrix[i][j] for j in indices_to_optimize] for i in indices_to_optimize])

    # Inisialisasi populasi (permutasi dari indeks lokasi yang akan dikunjungi)
    population = [random.sample(location_indices, len(location_indices)) for _ in range(pop_size)]
    
    convergence_data = []

    for gen in range(generations):
        # Hitung fitness untuk setiap individu di populasi
        fitness_values = []
        for route in population:
            # Rute lengkap: [start] -> [rute_optimasi]
            current_route = [start_index] + route
            total_travel_time = sum(full_time_matrix[current_route[i]][current_route[i+1]] for i in range(len(current_route) - 1))
            total_visit_time = locations_df['Estimasi Durasi Kunjungan (menit)'].sum()
            total_time = total_travel_time + total_visit_time
            fitness_values.append(1 / total_time if total_time > 0 else 0)

        # Simpan data untuk grafik konvergensi
        best_fitness_idx = np.argmax(fitness_values)
        best_time_minutes = 1 / fitness_values[best_fitness_idx]
        convergence_data.append(best_time_minutes / 60) # dalam jam

        # Buat populasi baru
        new_population = []

        # Elitisme: Bawa individu terbaik ke generasi berikutnya
        sorted_population = [p for _, p in sorted(zip(fitness_values, population), key=lambda x: x[0], reverse=True)]
        new_population.extend(sorted_population[:elitism_size])

        # Hasilkan sisa populasi dengan Crossover dan Mutasi
        for _ in range(pop_size - elitism_size):
            parent1 = selection(population, fitness_values)
            parent2 = selection(population, fitness_values)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)
        
        population = new_population

    # Dapatkan rute terbaik dari populasi final
    final_fitness_values = [fitness(route, time_matrix_for_ga, durations_to_optimize) for route in population]
    best_route_indices = population[np.argmax(final_fitness_values)]
    
    # Gabungkan dengan titik awal untuk mendapatkan rute final
    final_route_order = [start_index] + best_route_indices
    
    return final_route_order, full_time_matrix, convergence_data


# --- TAMPILAN APLIKASI STREAMLIT ---

st.set_page_config(page_title="Padang Tourism Route Optimizer", layout="wide")

# Custom CSS untuk tema gelap
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    .st-emotion-cache-16txtl3 { padding: 2rem 1rem 10rem; }
    .stButton>button { background-color: #0068c9; color: white; border-radius: 8px; }
    .stButton>button:hover { background-color: #00509d; color: white; }
    h1, h2, h3 { color: #fafafa; border-bottom: 2px solid #0068c9; padding-bottom: 0.3rem; }
    .st-b7 { color: #fafafa; }
    .stDataFrame { background-color: #1f2029; }
    .stMetric { background-color: #1f2029; border-radius: 8px; padding: 1rem; }
</style>
""", unsafe_allow_html=True)

st.title("🛣️ Padang Tourism Route Optimizer")
st.markdown("Optimalkan rute perjalanan wisata Anda di Kota Padang dengan Algoritma Genetika.")

# Muat data
try:
    data = pd.read_csv("data_tempat_wisata_padang.csv")
except FileNotFoundError:
    st.warning("File `data_tempat_wisata_padang.csv` tidak ditemukan. Mengambil data baru dari OpenStreetMap.")
    data = fetch_osm_data()

if st.button("🔄 Ambil Ulang Data dari OpenStreetMap"):
    data = fetch_osm_data()

# --- SIDEBAR PENGATURAN ---
st.sidebar.header("⚙️ Konfigurasi Perjalanan")
st.sidebar.markdown("---")

st.sidebar.subheader("📍 Pilih Tempat Wisata")
place_types = st.sidebar.multiselect(
    "Filter Berdasarkan Jenis Tempat",
    options=data['Jenis Tempat'].unique(),
    default=data['Jenis Tempat'].unique()[:4]
)

filtered_data = data[data['Jenis Tempat'].isin(place_types)]
all_place_names = filtered_data['Nama Tempat Wisata'].tolist()

selected_places = st.sidebar.multiselect(
    "Pilih Tempat yang Ingin Dikunjungi",
    options=all_place_names,
    default=all_place_names[:min(7, len(all_place_names))]
)

# --- MAIN CONTENT ---
if len(selected_places) < 2:
    st.info("ℹ️ Silakan pilih minimal 2 tempat wisata di sidebar untuk memulai optimasi rute.")
else:
    # Lanjutkan dengan pengaturan jika tempat sudah dipilih
    selected_data = filtered_data[filtered_data['Nama Tempat Wisata'].isin(selected_places)].reset_index(drop=True)
    
    start_place_name = st.sidebar.selectbox(
        "Pilih Lokasi Awal",
        options=selected_places
    )
    start_index = selected_data[selected_data['Nama Tempat Wisata'] == start_place_name].index[0]

    st.sidebar.markdown("---")
    st.sidebar.subheader("🚗 Parameter Perjalanan")
    speed_kmh = st.sidebar.slider("Kecepatan Rata-Rata (km/jam)", 20, 60, 30)
    start_time_input = st.sidebar.time_input("Waktu Mulai Perjalanan", value=datetime.strptime("08:00", "%H:%M").time())
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧠 Pengaturan Algoritma Genetika")
    pop_size = st.sidebar.slider("Ukuran Populasi", 50, 500, 100)
    generations = st.sidebar.slider("Jumlah Generasi", 100, 2000, 500)
    mutation_rate = st.sidebar.slider("Tingkat Mutasi", 0.01, 0.2, 0.05, 0.01)
    elitism_size = st.sidebar.slider("Jumlah Elit (Elitism)", 1, 10, 2)

    if st.sidebar.button("🚀 Hitung Rute Optimal", use_container_width=True):
        with st.spinner("Menghitung rute optimal menggunakan jaringan jalan raya... Ini mungkin butuh waktu."):
            
            # Jalankan Algoritma Genetika
            best_route_indices, time_matrix, convergence_data = genetic_algorithm(
                selected_data, start_index, pop_size, generations, mutation_rate, elitism_size, speed_kmh
            )
            
            # --- TAMPILKAN HASIL ---
            st.header("✨ Hasil Optimasi Rute")
            
            # 1. Tampilkan Metrik Utama
            itinerary_df = pd.DataFrame(columns=["No", "Lokasi", "Waktu Tiba", "Durasi Kunjungan (menit)", "Waktu Berangkat"])
            current_time = datetime.combine(datetime.today(), start_time_input)
            total_travel_time = 0

            for i, loc_index in enumerate(best_route_indices):
                place_info = selected_data.iloc[loc_index]
                visit_duration = place_info['Estimasi Durasi Kunjungan (menit)']
                
                if i > 0:
                    prev_loc_index = best_route_indices[i-1]
                    travel_time_minutes = time_matrix[prev_loc_index, loc_index]
                    total_travel_time += travel_time_minutes
                    current_time += timedelta(minutes=travel_time_minutes)
                
                arrival_time_str = current_time.strftime('%H:%M')
                departure_time = current_time + timedelta(minutes=visit_duration)
                departure_time_str = departure_time.strftime('%H:%M')
                
                itinerary_df.loc[i] = [i + 1, place_info['Nama Tempat Wisata'], arrival_time_str, f"{visit_duration:.0f}", departure_time_str]
                current_time = departure_time
            
            total_visit_time = selected_data['Estimasi Durasi Kunjungan (menit)'].sum()
            total_trip_time_hours = (total_travel_time + total_visit_time) / 60
            
            col1, col2, col3 = st.columns(3)
            col1.metric("⏱️ Total Waktu Perjalanan", f"{total_trip_time_hours:.2f} Jam")
            col2.metric("🚗 Waktu di Jalan", f"{total_travel_time:.0f} Menit")
            col3.metric("🏞️ Waktu di Lokasi", f"{total_visit_time:.0f} Menit")
            
            # 2. Tampilkan Jadwal (Itinerary)
            st.subheader("🗓️ Jadwal Perjalanan (Itinerary)")
            st.dataframe(itinerary_df.set_index('No'), use_container_width=True)
            
            # 3. Tampilkan Peta Rute
            st.subheader("🗺️ Peta Rute Optimal")
            map_center = [selected_data['latitude'].mean(), selected_data['longitude'].mean()]
            m = folium.Map(location=map_center, zoom_start=13)
            
            route_coords = [
                (selected_data.iloc[i]['latitude'], selected_data.iloc[i]['longitude']) for i in best_route_indices
            ]
            
            # Tambahkan Marker untuk setiap lokasi
            for i, loc_index in enumerate(best_route_indices):
                place_info = selected_data.iloc[loc_index]
                popup_content = f"<b>{i+1}. {place_info['Nama Tempat Wisata']}</b><br>Tiba: {itinerary_df.iloc[i]['Waktu Tiba']}"
                folium.Marker(
                    location=(place_info['latitude'], place_info['longitude']),
                    popup=folium.Popup(popup_content, max_width=300),
                    tooltip=f"Klik untuk detail: {place_info['Nama Tempat Wisata']}",
                    icon=folium.Icon(color='blue' if i == 0 else 'green', icon='info-sign' if i > 0 else 'play', prefix='glyphicon')
                ).add_to(m)

            # Tambahkan Garis Rute
            folium.PolyLine(route_coords, color='#ff7f0e', weight=5, opacity=0.8, tooltip="Rute Optimal").add_to(m)
            folium_static(m, width=None, height=500)

            # 4. Tampilkan Grafik Konvergensi
            st.subheader("📈 Grafik Konvergensi Algoritma")
            st.markdown("Grafik ini menunjukkan bagaimana total waktu perjalanan (dalam jam) menurun dan menjadi stabil seiring berjalannya generasi.")
            fig, ax = plt.subplots()
            ax.plot(range(generations), convergence_data, color='#0068c9', linewidth=2)
            ax.set_xlabel("Generasi")
            ax.set_ylabel("Total Waktu Terbaik (Jam)")
            ax.set_title("Konvergensi Algoritma Genetika")
            ax.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig)

# Menampilkan data mentah yang digunakan
st.subheader("📚 Daftar Tempat Wisata Terpilih")
st.dataframe(selected_data[['Nama Tempat Wisata', 'Jenis Tempat', 'Estimasi Durasi Kunjungan (menit)', 'latitude', 'longitude']], use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #a1a1a1; padding: 1rem;">
    © 2025 Padang Tourism Route Optimizer | Dibuat dengan Streamlit & OSMnx
</div>
""", unsafe_allow_html=True)