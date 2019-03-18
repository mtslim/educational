public class SimpleDotComTestDrive {
	public static void main (String[] args) {
		
		SimpleDotCom dot = new SimpleDotCom(); // instantiate a SimpleDotCom object
		
		int[] locations = {2, 3, 4}; // make an int array for the location of the dot com
		dot.setLocationCells(locations); // invoke the setter method on the dot com
		
		String userGuess = "2";
		String result = dot.checkYourself(userGuess); // invoke the checkYourself method
	}	
}