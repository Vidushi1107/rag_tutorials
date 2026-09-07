# Helix Robotics — Support FAQ

## Error codes

**E-101 — Safety LIDAR obstructed.** A safety scanner is blocked or dirty. Clean both scanner
windows with a dry microfibre cloth. If the code persists on a clean scanner, the scanner
module has likely failed and is covered under warranty.

**E-204 — Localisation lost.** The robot cannot match its surroundings to the stored map. This
usually follows a layout change. Re-run the mapping routine from HelixOS under
**Fleet → Maps → Rescan**. Note that E-204 is the single most common support ticket Helix
receives, and roughly 70% of cases are caused by racking moved without a map update.

**E-311 — Charge fault.** The robot docked but did not draw current. Check the dock contacts
for debris, then confirm the dock is a Beacon D2 for Atlas robots or a Beacon D1 for Scout
robots. Cross-docking an Atlas onto a D1 will always produce E-311.

**E-450 — Payload over limit.** The load exceeds the rated payload. For the Atlas R5 this is
450 kg. The robot will refuse to move until the load is reduced.

**E-502 — Firmware mismatch.** The robot's firmware branch is incompatible with the HelixOS
version. Upgrade HelixOS first, then the robot. Never downgrade robot firmware; downgrades are
not supported and can brick the drive controller.

## Common questions

**How do I update firmware?**
Firmware is pushed from HelixOS under **Fleet → Firmware**. Atlas robots must be docked and
above 40% charge; the update takes about 25 minutes. Scout robots can update undocked above
60% charge and take about 11 minutes.

**Can Atlas and Scout robots share a charging dock?**
No. Atlas uses the Beacon D2 dock and Scout uses the Beacon D1. The connectors are physically
different and the voltages do not match.

**What network does HelixOS need?**
A dedicated 5 GHz WLAN with roaming enabled, minimum -65 dBm signal everywhere the robots
travel, and less than 50 ms latency to the HelixOS server. 2.4 GHz is not supported for robot
traffic.

**How many robots can one HelixOS instance manage?**
A single instance supports up to 120 robots. Beyond that, Helix deploys a second instance with
a federation gateway.

**What happens during a network outage?**
Robots complete their current task, then park safely and wait. They do not accept new tasks
until HelixOS connectivity is restored. Robots hold their last known map locally, so
localisation survives an outage.

**Is there an API?**
Yes. HelixOS 4.x exposes a REST API and a WebSocket telemetry stream. API access requires the
Integration add-on, priced at €4,800 per site per year.
