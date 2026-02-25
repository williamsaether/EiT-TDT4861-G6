import requests
import time
from pyproj import Transformer
import json

# ================= Konfigurasjon =================
NVDB_BASE_URL = "https://nvdbapiles.atlas.vegvesen.no"
NVDB_POSISJON_URL = f"{NVDB_BASE_URL}/vegnett/api/v4/posisjon"
NVDB_OBJEKT_URL = f"{NVDB_BASE_URL}/vegobjekter/api/v4/vegobjekter/105"

NVDB_HEADERS = {
    "X-Client": "NTNU_EiT_StudentProject_Final_v4",
    "Accept": "application/vnd.vegvesen.nvdb-v4+json"
}

transformer = Transformer.from_crs("EPSG:4326", "EPSG:5973", always_xy=True)

def log_debug(step, url, params, resp):
    print(f"\n--- DEBUG: {step} ---")
    print(f"URL: {url}")
    print(f"Status: {resp.status_code}")
    try:
        data = resp.json()
        print(f"Body: {json.dumps(data, indent=2, ensure_ascii=False)[:800]}...")
    except:
        print(f"Body: {resp.text[:200]}")

# ================= Funksjoner =================

def get_robust_speed_limit(lat, lon):
    ost, nord = transformer.transform(lon, lat)
    
    # --- STEG 1: FINN VEI ---
    pos_params = {
        "nord": nord, 
        "ost": ost,
        "srid": 5973, 
        "maks_avstand": 150,  # ✅ Økt til 150m
        "maks_antall": 1
    }
    
    try:
        pos_resp = requests.get(NVDB_POSISJON_URL, params=pos_params, headers=NVDB_HEADERS, timeout=10)
        log_debug("POSISJON (SNAPPING)", NVDB_POSISJON_URL, pos_params, pos_resp)
        
        pos_data = pos_resp.json()
        if not pos_data:
            return "Ingen vei funnet i nærheten (prøv å øke maks_avstand)"
        
        match = pos_data[0]
        vsys = match.get('vegsystemreferanse', {})
        
        veikategori = vsys.get('vegsystem', {}).get('vegkategori')
        veinummer = vsys.get('vegsystem', {}).get('nummer')
        kortform_full = vsys.get('kortform') or "Ukjent vei"
        avstand = match.get('avstand', 'ukjent')
        
        print(f"\n✅ Funnet vei: {kortform_full} ({avstand}m unna)")
        print(f"   Veikategori: {veikategori}, Nummer: {veinummer}")

        # --- STEG 2: HENT FART (Geografisk søk) ---
        size = 50
        kartutsnitt = f"{ost-size},{nord-size},{ost+size},{nord+size}"
        
        obj_params = {
            "kartutsnitt": kartutsnitt,
            "srid": 5973,
            "inkluder": "egenskaper,lokasjon",
            "antall": 20
        }
        
        obj_resp = requests.get(NVDB_OBJEKT_URL, params=obj_params, headers=NVDB_HEADERS, timeout=10)
        log_debug("VEGOBJEKT (FART)", NVDB_OBJEKT_URL, obj_params, obj_resp)
        
        if obj_resp.status_code == 200:
            obj_data = obj_resp.json()
            objekter = obj_data.get("objekter", [])
            
            if not objekter:
                return f"Fant vei ({kortform_full}), men ingen fartsgrense i området"

            # ✅ FIKSET: Iterer gjennom objekter og finn matching
            for obj in objekter:
                # Hent fartsgrense-verdien FRA OBJEKTET
                fart_verdi = None
                for e in obj.get("egenskaper", []):
                    if e["id"] == 2021:
                        fart_verdi = e["verdi"]  # ✅ DETTE ER DEN FAKTISKE VERDIEN
                        break
                
                if not fart_verdi:
                    continue
                
                # Sjekk om dette objektet matcher veien
                match_found = False
                
                if veikategori and veinummer:
                    for vref in obj.get("lokasjon", {}).get("vegsystemreferanser", []):
                        v_sys = vref.get("vegsystem", {})
                        if (v_sys.get("vegkategori") == veikategori and 
                            v_sys.get("nummer") == veinummer):
                            match_found = True
                            # ✅ FIKSET: Print faktisk verdi, ikke hardkodet
                            print(f"✅ Match funnet: {veikategori}{veinummer} = {fart_verdi} km/t")
                            break
                else:
                    # For vei uten nummer
                    match_found = True
                
                if match_found:
                    # ✅ KRITISK FIX: Returner fart_verdi, IKKE hardkodet streng
                    return f"{kortform_full}: {fart_verdi} km/t"
            
            # Fallback: ta første fartsgrense
            for obj in objekter:
                for e in obj.get("egenskaper", []):
                    if e["id"] == 2021:
                        fart = e["verdi"]  # ✅ FIKSET
                        print(f"⚠️  Bruker nærmeste: {fart} km/t")
                        return f"{kortform_full} (Nærmeste): {fart} km/t"
        
        return f"Fant vei ({kortform_full}), men ingen fartsgrense"

    except Exception as e:
        print(f"\n❌ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        return f"Systemfeil: {str(e)}"

# ================= Kjøring =================

def main():
    points = {
        # "Tømmerdalsveien": (63.435512, 10.275317),
        # "E6": (63.326244,10.334259),
        "E6 (80)": (63.333542, 10.356348)
    }

    for name, (lat, lon) in points.items():
        print(f"\n{'#'*60}")
        print(f" ANALYSERER: {name}")
        print(f"{'#'*60}")
        resultat = get_robust_speed_limit(lat, lon)
        print(f"\n🏁 RESULTAT FOR {name}: {resultat}")
        time.sleep(1)

if __name__ == "__main__":
    main()