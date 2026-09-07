# Atlas R5 — Product Specification

The Atlas R5 is Helix Robotics' flagship heavy-payload autonomous mobile robot, released in
March 2023. It replaced the Atlas R4, which was discontinued in December 2023.

## Physical specifications

| Property | Value |
| --- | --- |
| Maximum payload | 450 kg |
| Unloaded weight | 210 kg |
| Dimensions (L x W x H) | 1,340 x 780 x 340 mm |
| Maximum speed (laden) | 1.6 m/s |
| Maximum speed (unladen) | 2.4 m/s |
| Turning radius | Zero (differential drive) |
| Ingress protection | IP54 |
| Operating temperature | -5 °C to 40 °C |

## Power

- **Battery:** 48 V lithium iron phosphate (LiFePO4), 2.1 kWh
- **Runtime:** 9.5 hours under typical mixed load
- **Charge time:** 42 minutes to 80%, 71 minutes to full
- **Charging:** Opportunity charging via Beacon D2 dock; the robot returns to dock automatically
  below 25% state of charge
- **Battery service life:** 2,800 full cycles to 80% retained capacity

## Sensing and navigation

- Dual 2D safety LIDAR (front and rear), 270° coverage each
- Forward stereo depth camera for obstacle classification
- Six ultrasonic proximity sensors
- Wheel odometry plus IMU for dead reckoning
- Navigation uses natural-feature SLAM; reflective markers are **not** required

The Atlas R5 is certified to ISO 3691-4 and carries CE marking. It is **not** rated for outdoor
use or for cold-store environments below -5 °C. Customers running cold-store sites should
specify the Atlas R5-C variant instead, which is rated to -25 °C and carries a 12% price premium.

## Pricing

- **Atlas R5 base unit:** €78,000
- **Atlas R5-C (cold store):** €87,360
- **Beacon D2 charging dock:** €6,400
- Volume discount of 8% applies at 10 units, 14% at 25 units.

## Firmware

The Atlas R5 runs Helix firmware branch `atlas-5.x`. Firmware updates are delivered through
HelixOS and require the robot to be docked and above 40% charge. A firmware update takes
approximately 25 minutes, during which the robot is unavailable for tasks.
