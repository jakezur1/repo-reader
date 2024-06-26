const path = require("path");
const HTMLPlugin = require("html-webpack-plugin");
const CopyPlugin = require("copy-webpack-plugin")

module.exports = {
	entry: {
		index: "/src/index.tsx",
		background: path.resolve('background.js')
	},
	mode: "production",
	module: {
		rules: [
			{
				test: /\.tsx?$/,
				use: [
					{
						loader: "ts-loader",
						options: {
							compilerOptions: {noEmit: false},
						}
					}],
				exclude: /node_modules/,
			},
			{
				exclude: /node_modules/,
				test: /\.css$/i,
				use: [
					"style-loader",
					"css-loader",
					'postcss-loader'
				]
			},
		],
	},
	plugins: [
		new CopyPlugin({
			patterns: [
				{from: "./public/manifest.json", to: "./manifest.json"},
			],
		}),
		...getHtmlPlugins(["index"]),
	],
	resolve: {
		extensions: [".tsx", ".ts", ".js", ".jsx"],
	},
	output: {
		path: path.join(__dirname, "dist/js"),
		filename: "[name].js",
	},
};

function getHtmlPlugins(chunks) {
	return chunks.map(
		(chunk) =>
			new HTMLPlugin({
				title: "React extension",
				filename: `${chunk}.html`,
				chunks: [chunk],
			})
	);
}
