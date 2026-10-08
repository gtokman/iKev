import type { Metadata, Viewport } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'iKev — Small model. Clear decisions.',
  description: 'Run Kev decision models locally on Apple devices. A native Swift package powered by MLX. Fixed choices, calibrated probabilities, no token generation.',
  manifest: '/manifest.json',
};
export const viewport: Viewport = { width: 'device-width', initialScale: 1, maximumScale: 1, themeColor: '#101110' };
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return <html lang="en"><body>{children}</body></html>;
}
