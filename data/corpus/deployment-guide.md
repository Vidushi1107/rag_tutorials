# Helix Robotics — Deployment Guide

This guide covers what a site must provide before Helix robots can be commissioned.

## 1. Site survey

Every deployment begins with a site survey conducted by a Helix field engineer. The survey
takes 1-2 days for sites under 10,000 m² and 3-5 days above that. The survey produces a
floor-flatness report, a WLAN heat map, and a traffic plan.

Site surveys are chargeable at €2,200 per day but are credited back in full against the order
if a purchase follows within 6 months.

## 2. Floor requirements

- Floor flatness must meet **DIN 15185** for AMR traffic lanes.
- Maximum step or joint height the Atlas R5 can cross: 15 mm.
- Maximum sustained gradient: 4% laden, 6% unladen.
- Floors must be sealed concrete, epoxy, or equivalent. Bare or dusty concrete causes wheel
  slip and accelerates tread wear, which is a consumable and not covered by warranty.
- Drainage channels wider than 20 mm must be bridged or excluded from the traffic plan.

## 3. Network requirements

- Dedicated 5 GHz WLAN SSID for robot traffic; 2.4 GHz is not supported.
- Signal strength no weaker than -65 dBm anywhere in the operating area.
- Fast roaming (802.11r) enabled, with roaming handover under 150 ms.
- Latency to the HelixOS server under 50 ms, packet loss under 0.5%.
- HelixOS requires outbound HTTPS on port 443 for telemetry and firmware delivery.

## 4. Power and docks

Each Beacon dock needs a dedicated 16 A circuit. Helix recommends one dock per three Atlas
robots for single-shift operations and one dock per two Atlas robots for continuous
multi-shift operations. Scout deployments can run one D1 dock per four robots.

Docks must sit against a wall with 1.5 m of clear approach and must not be placed in a main
traffic aisle.

## 5. Commissioning

Commissioning is performed by Helix and takes 2-4 days depending on fleet size. Steps:

1. Map the facility with the mapping trolley or a robot in manual mode.
2. Define traffic lanes, keep-out zones, and pick/drop stations in HelixOS.
3. Run a supervised shakedown of at least 40 task cycles per robot.
4. Train site staff — a mandatory 4-hour operator course plus an optional 8-hour
   supervisor course.
5. Sign commissioning off. **The warranty period starts at sign-off.**

## 6. Ongoing site obligations

If racking, walls, or fixed equipment move, the site must re-run the mapping routine before
resuming automated traffic. Skipping this step is the leading cause of E-204 localisation
faults. Helix does not treat map drift caused by unreported layout changes as a warranty
issue.
