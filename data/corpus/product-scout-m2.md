# Scout M2 — Product Specification

The Scout M2 is a lightweight inventory-scanning robot for cycle counting, slot verification,
and stock audits. It was released in September 2024 and supersedes the Scout M1.

## Physical specifications

| Property | Value |
| --- | --- |
| Maximum payload | 12 kg |
| Unloaded weight | 34 kg |
| Dimensions (L x W x H) | 620 x 480 x 1,850 mm (mast extended) |
| Mast height range | 900 mm to 1,850 mm |
| Maximum speed | 1.8 m/s |
| Ingress protection | IP42 |
| Operating temperature | 0 °C to 40 °C |

The Scout M2 is a scanning platform, not a transport robot. The 12 kg payload rating covers
auxiliary sensors and label printers only; it must not be used to move goods.

## Power

- **Battery:** 24 V lithium-ion, 0.6 kWh
- **Runtime:** 6 hours of continuous scanning
- **Charge time:** 55 minutes to full
- **Charging:** Beacon D1 dock (the Scout M2 is **not** compatible with the Beacon D2 dock used
  by Atlas robots)

## Scanning hardware

- Telescoping mast with four RFID antennas (UHF, EPC Gen2)
- Two 20 MP barcode/label cameras with adaptive illumination
- Reads up to 1,100 tags per minute at 4 m read range
- Optional thermal camera module (+€3,200) for cold-chain spot checks

## Accuracy

In Helix's reference deployments the Scout M2 achieves 99.2% inventory read accuracy on
UHF-tagged stock and 97.6% on barcode-only stock. Accuracy degrades on metal-dense racking;
Helix recommends a site survey before committing to barcode-only deployments.

## Pricing

- **Scout M2 base unit:** €19,500
- **Beacon D1 charging dock:** €2,900
- **Thermal camera module:** €3,200
- Volume discount of 8% applies at 10 units, 14% at 25 units.

## Firmware

The Scout M2 runs Helix firmware branch `scout-2.x`. Unlike the Atlas line, Scout firmware
updates can be applied while undocked provided charge is above 60%. An update takes
approximately 11 minutes.
