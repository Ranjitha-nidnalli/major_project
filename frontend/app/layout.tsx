import type { Metadata } from 'next'
import { Geist, Geist_Mono } from 'next/font/google'
import { Analytics } from '@vercel/analytics/next'
import './globals.css'

const _geist = Geist({ subsets: ["latin"] });
const _geistMono = Geist_Mono({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: 'ಕೃಷಿ ಸಹಾಯಕ - Karnataka Sugarcane Farming AI Assistant',
  description: 'ಕರ್ನಾಟಕ ಕಬ್ಬು ಬೆಳೆಗಾರರಿಗಾಗಿ AI ಕೃಷಿ ಸಹಾಯಕ - Sugarcane farming assistance in Kannada. Other crops are planned but not yet supported.',
  generator: 'v0.app',
  icons: {
    icon: [
      {
        url: '/icon-light-32x32.png',
        media: '(prefers-color-scheme: light)',
      },
      {
        url: '/icon-dark-32x32.png',
        media: '(prefers-color-scheme: dark)',
      },
      {
        url: '/icon.svg',
        type: 'image/svg+xml',
      },
    ],
    apple: '/apple-icon.png',
  },
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="kn">
      <body className="font-sans antialiased">
        {children}
        <Analytics />
      </body>
    </html>
  )
}
