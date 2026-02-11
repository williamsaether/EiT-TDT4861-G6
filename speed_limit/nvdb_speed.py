import requests
from pyproj import Transformer

# --- Konfigurasjon ---
NVDB_BASE_URL = "https://nvdbapiles.atlas.vegvesen.no"
NVDB_POSISJON_URL = f"{NVDB_BASE_URL}/vegnett/api/v4/posisjon"
NVDB_OBJEKT_URL = f"{NVDB_BASE_URL}/vegobjekter/api/v4/vegobjekter/105"

NVDB_HEADERS = {
    "X-Client": "NTNU_EiT_StudentProject_Final_v4",
    "Accept": "application/vnd.vegvesen.nvdb-v4+json"
}

# Transformator fra GPS (WGS84) til NVDB (UTM33)
transformer = Transformer.from_crs("EPSG:4326", "EPSG:5973", always_xy=True)

def get_speed_limit_data(lat, lon):
    """
    Henter fartsgrense og veiinformasjon for en gitt koordinat.
    Returnerer en dict med data eller None hvis ikke funnet.
    """
    ost, nord = transformer.transform(lon, lat)
    
    pos_params = {
        "nord": nord, 
        "ost": ost,
        "srid": 5973, 
        "maks_avstand": 40,
        "maks_antall": 5,
        "trafikantgruppe": "K"
    }
    
    try:
        pos_resp = requests.get(NVDB_POSISJON_URL, params=pos_params, headers=NVDB_HEADERS, timeout=10)
        pos_data = pos_resp.json()
        
        if not pos_data:
            return None

        for match in pos_data:
            vsys = match.get('vegsystemreferanse', {})
            veg_navn = vsys.get('kortform', 'Ukjent vei')
            vls = match.get('veglenkesekvens', {})
            vls_id = vls.get('veglenkesekvensid')
            rel_pos = vls.get('relativPosisjon')
            distanse = match.get('avstand', 0)

            obj_params = {
                "veglenkesekvens": f"{rel_pos}@{vls_id}",
                "inkluder": "egenskaper",
                "srid": 5973
            }
            
            obj_resp = requests.get(NVDB_OBJEKT_URL, params=obj_params, headers=NVDB_HEADERS, timeout=10)
            
            if obj_resp.status_code == 200:
                obj_list = obj_resp.json().get("objekter", [])
                if obj_list:
                    fart = None
                    for e in obj_list[0].get("egenskaper", []):
                        if e["id"] == 2021:
                            fart = e["verdi"]
                            break
                    
                    if fart:
                        return {
                            "status": "ok",
                            "fartsgrense": int(fart),
                            "vei": veg_navn,
                            "avstand_meter": round(distanse, 1),
                            "full_info": f"{fart} km/t på {veg_navn}"
                        }
        return {"status":"error", "message": "Ingen fartsgrense funnet i nærheten"}
    except Exception as e:
        return {"status":"error", "message": str(e)}