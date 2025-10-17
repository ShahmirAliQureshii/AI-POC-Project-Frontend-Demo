module.exports = {
  reactStrictMode: true,
  images: {
    domains: ['your-image-domain.com'], // Replace with your image domain if needed
  },
  env: {
    API_URL: process.env.API_URL || 'http://localhost:3000/api/recognition', // Set your API URL
  },
};